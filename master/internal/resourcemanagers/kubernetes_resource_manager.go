package resourcemanagers

import (
	"fmt"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"

	"github.com/determined-ai/determined/master/internal/config"
	"github.com/determined-ai/determined/master/internal/job"
	"github.com/determined-ai/determined/master/internal/resourcemanagers/kubernetes"
	"github.com/determined-ai/determined/master/internal/sproto"
	"github.com/determined-ai/determined/master/pkg/actor"
	"github.com/determined-ai/determined/master/pkg/actor/actors"
	"github.com/determined-ai/determined/master/pkg/aproto"
	"github.com/determined-ai/determined/master/pkg/cproto"
	"github.com/determined-ai/determined/master/pkg/device"
	"github.com/determined-ai/determined/master/pkg/logger"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/master/pkg/tasks"
	"github.com/determined-ai/determined/proto/pkg/apiv1"
	"github.com/determined-ai/determined/proto/pkg/resourcepoolv1"
)

// KubernetesDummyResourcePool is the name of the dummy resource pool for kubernetes.
const KubernetesDummyResourcePool = "kubernetes"
const kubernetesScheduler = "kubernetes"

// kubernetesResourceProvider manages the lifecycle of k8s resources.
type kubernetesResourceManager struct {
	config *config.KubernetesResourceManagerConfig

	reqList           *taskList
	groups            map[*actor.Ref]*group
	addrToContainerID map[*actor.Ref]cproto.ID
	containerIDtoAddr map[string]*actor.Ref
	slotsUsedPerGroup map[*group]int

	podsActor *actor.Ref

	reschedule bool

	queuePositions  jobSortState
	echoRef         *echo.Echo
	masterTLSConfig model.TLSClientConfig
	loggingConfig   model.LoggingConfig
}

func newKubernetesResourceManager(
	config *config.KubernetesResourceManagerConfig,
	echoRef *echo.Echo,
	masterTLSConfig model.TLSClientConfig,
	loggingConfig model.LoggingConfig,
) actor.Actor {
	return &kubernetesResourceManager{
		config: config,

		reqList:           newTaskList(),
		groups:            make(map[*actor.Ref]*group),
		addrToContainerID: make(map[*actor.Ref]cproto.ID),
		containerIDtoAddr: make(map[string]*actor.Ref),
		slotsUsedPerGroup: make(map[*group]int),
		queuePositions:    initalizeJobSortState(),

		echoRef:         echoRef,
		masterTLSConfig: masterTLSConfig,
		loggingConfig:   loggingConfig,
	}
}

func (k *kubernetesResourceManager) Receive(ctx *actor.Context) error {
	reschedule := true
	defer func() {
		// Default to scheduling every 500ms if a message was received, but allow messages
		// that don't affect the cluster to be skipped.
		k.reschedule = k.reschedule || reschedule
	}()

	switch msg := ctx.Message().(type) {
	case actor.PreStart:
		actors.NotifyAfter(ctx, actionCoolDown, schedulerTick{})

		k.podsActor = kubernetes.Initialize(
			ctx.Self().System(),
			k.echoRef,
			ctx.Self(),
			k.config.Namespace,
			k.config.MasterServiceName,
			k.masterTLSConfig,
			k.loggingConfig,
			k.config.LeaveKubernetesResources,
			k.config.DefaultScheduler,
			k.config.SlotType,
			kubernetes.PodSlotResourceRequests{CPU: k.config.SlotResourceRequests.CPU},
			k.config.Fluent,
		)

	case
		groupActorStopped,
		sproto.SetGroupMaxSlots,
		sproto.SetTaskName,
		sproto.AllocateRequest,
		sproto.ResourcesReleased,
		sproto.UpdatePodStatus:
		return k.receiveRequestMsg(ctx)

	case
		job.GetJobQ,
		job.GetJobSummary,
		job.GetJobQStats,
		job.SetGroupWeight,
		job.SetGroupPriority,
		job.MoveJob,
		*apiv1.GetJobQueueStatsRequest:
		return k.receiveJobQueueMsg(ctx)

	case sproto.GetTaskHandler:
		reschedule = false
		ctx.Respond(getTaskHandler(k.reqList, msg.ID))

	case sproto.GetTaskSummary:
		if resp := getTaskSummary(k.reqList, *msg.ID, k.groups, kubernetesScheduler); resp != nil {
			ctx.Respond(*resp)
		}
		reschedule = false

	case sproto.GetTaskSummaries:
		reschedule = false
		ctx.Respond(getTaskSummaries(k.reqList, k.groups, kubernetesScheduler))

	case *apiv1.GetResourcePoolsRequest:
		resourcePoolSummary, err := k.summarizeDummyResourcePool(ctx)
		if err != nil {
			ctx.Respond(err)
		}
		resp := &apiv1.GetResourcePoolsResponse{
			ResourcePools: []*resourcepoolv1.ResourcePool{resourcePoolSummary},
		}
		ctx.Respond(resp)

	case sproto.GetDefaultComputeResourcePoolRequest:
		ctx.Respond(sproto.GetDefaultComputeResourcePoolResponse{PoolName: ""})

	case sproto.GetDefaultAuxResourcePoolRequest:
		ctx.Respond(sproto.GetDefaultAuxResourcePoolResponse{PoolName: ""})

	case sproto.ValidateCommandResourcesRequest:
		fulfillable := k.config.MaxSlotsPerPod >= msg.Slots
		ctx.Respond(sproto.ValidateCommandResourcesResponse{Fulfillable: fulfillable})

	case schedulerTick:
		if k.reschedule {
			k.schedulePendingTasks(ctx)
		}
		k.reschedule = false
		reschedule = false
		actors.NotifyAfter(ctx, actionCoolDown, schedulerTick{})
	case *apiv1.GetAgentsRequest:
		resp := ctx.Ask(k.podsActor, msg)
		ctx.Respond(resp.Get())
	default:
		reschedule = false
		ctx.Log().Errorf("unexpected message %T", msg)
		return actor.ErrUnexpectedMessage(ctx)
	}

	return nil
}

func (k *kubernetesResourceManager) summarizeDummyResourcePool(
	ctx *actor.Context,
) (*resourcepoolv1.ResourcePool, error) {
	slotsUsed := 0
	for _, slotsUsedByGroup := range k.slotsUsedPerGroup {
		slotsUsed += slotsUsedByGroup
	}

	pods, err := k.summarizePods(ctx)
	if err != nil {
		return nil, err
	}

	// Expose a fake number of zero slots here just to signal to the UI
	// that this RP does support the aux containers.

	return &resourcepoolv1.ResourcePool{
		Name:                         KubernetesDummyResourcePool,
		Description:                  "Kubernetes-managed pool of resources",
		Type:                         resourcepoolv1.ResourcePoolType_RESOURCE_POOL_TYPE_K8S,
		NumAgents:                    int32(pods.NumAgents),
		SlotType:                     k.config.SlotType.Proto(),
		SlotsAvailable:               int32(pods.SlotsAvailable),
		SlotsUsed:                    int32(slotsUsed),
		AuxContainerCapacity:         int32(1),
		AuxContainersRunning:         int32(0),
		DefaultComputePool:           true,
		DefaultAuxPool:               true,
		Preemptible:                  k.config.GetPreemption(),
		MinAgents:                    0,
		MaxAgents:                    0,
		SlotsPerAgent:                int32(k.config.MaxSlotsPerPod),
		AuxContainerCapacityPerAgent: int32(1),
		SchedulerType:                resourcepoolv1.SchedulerType_SCHEDULER_TYPE_KUBERNETES,
		SchedulerFittingPolicy:       resourcepoolv1.FittingPolicy_FITTING_POLICY_KUBERNETES,
		Location:                     "kubernetes",
		ImageId:                      "",
		InstanceType:                 "kubernetes",
		Details:                      &resourcepoolv1.ResourcePoolDetail{},
	}, nil
}

func (k *kubernetesResourceManager) summarizePods(
	ctx *actor.Context,
) (*kubernetes.PodsInfo, error) {
	resp := ctx.Ask(k.podsActor, kubernetes.SummarizeResources{})
	if err := resp.Error(); err != nil {
		return nil, err
	}
	pods, ok := resp.Get().(*kubernetes.PodsInfo)
	if !ok {
		return nil, actor.ErrUnexpectedMessage(ctx)
	}
	return pods, nil
}

func (k *kubernetesResourceManager) receiveRequestMsg(ctx *actor.Context) error {
	switch msg := ctx.Message().(type) {
	case groupActorStopped:
		delete(k.slotsUsedPerGroup, k.groups[msg.Ref])
		delete(k.groups, msg.Ref)

	case sproto.SetGroupMaxSlots:
		k.getOrCreateGroup(ctx, msg.Handler).maxSlots = msg.MaxSlots

	case sproto.SetTaskName:
		k.receiveSetTaskName(ctx, msg)

	case sproto.AllocateRequest:
		k.addTask(ctx, msg)

	case sproto.ResourcesReleased:
		k.resourcesReleased(ctx, msg.TaskActor)

	case sproto.UpdatePodStatus:
		var ref *actor.Ref
		if addr, ok := k.containerIDtoAddr[msg.ContainerID]; ok {
			ref = addr
		}

		for it := k.reqList.iterator(); it.next(); {
			req := it.value()
			if req.TaskActor == ref {
				req.State = msg.State
			}
		}

	default:
		return actor.ErrUnexpectedMessage(ctx)
	}
	return nil
}

func (k *kubernetesResourceManager) addTask(ctx *actor.Context, msg sproto.AllocateRequest) {
	actors.NotifyOnStop(ctx, msg.TaskActor, sproto.ResourcesReleased{TaskActor: msg.TaskActor})

	if len(msg.AllocationID) == 0 {
		msg.AllocationID = model.AllocationID(uuid.New().String())
	}
	if msg.Group == nil {
		msg.Group = msg.TaskActor
	}
	k.getOrCreateGroup(ctx, msg.Group)
	if len(msg.Name) == 0 {
		msg.Name = "Unnamed-k8-Task"
	}

	ctx.Log().Infof(
		"resources are requested by %s (Allocation ID: %s)",
		msg.TaskActor.Address(), msg.AllocationID,
	)
	k.reqList.AddTask(&msg)
}

func (k *kubernetesResourceManager) receiveJobQueueMsg(ctx *actor.Context) error {
	switch msg := ctx.Message().(type) {
	case job.GetJobQ:
		ctx.Respond(k.jobQInfo())

	case *apiv1.GetJobQueueStatsRequest:
		resp := &apiv1.GetJobQueueStatsResponse{
			Results: make([]*apiv1.RPQueueStat, 0),
		}
		resp.Results = append(resp.Results, &apiv1.RPQueueStat{
			Stats:        jobStats(k.reqList),
			ResourcePool: KubernetesDummyResourcePool},
		)
		ctx.Respond(resp)

	case job.GetJobQStats:
		ctx.Respond(jobStats(k.reqList))

	case job.SetGroupWeight:
		// setting weights in kubernetes is not supported

	case job.SetGroupPriority:
		group := k.getOrCreateGroup(ctx, msg.Handler)
		group.priority = &msg.Priority

		for it := k.reqList.iterator(); it.next(); {
			if it.value().Group == msg.Handler {
				taskActor := it.value().TaskActor
				if id, ok := k.addrToContainerID[taskActor]; ok {
					ctx.Tell(k.podsActor, kubernetes.ChangePriority{PodID: id})
					delete(k.addrToContainerID, taskActor)
				}
			}
		}

	case job.MoveJob:
		ctx.Respond(fmt.Errorf("modifying job positions is not yet supported in Kubernetes"))

	default:
		return actor.ErrUnexpectedMessage(ctx)
	}
	return nil
}

func (k *kubernetesResourceManager) jobQInfo() map[model.JobID]*job.RMJobInfo {
	reqs := sortTasksWithPosition(k.reqList, k.groups, k.queuePositions, true)
	jobQinfo, _ := reduceToJobQInfo(reqs)

	return jobQinfo
}

func (k *kubernetesResourceManager) receiveSetTaskName(ctx *actor.Context, msg sproto.SetTaskName) {
	if task, found := k.reqList.GetTaskByHandler(msg.TaskHandler); found {
		task.Name = msg.Name
	}
}

func (k *kubernetesResourceManager) assignResources(
	ctx *actor.Context, req *sproto.AllocateRequest,
) {
	numPods := 1
	slotsPerPod := req.SlotsNeeded
	if req.SlotsNeeded > 1 {
		if k.config.MaxSlotsPerPod == 0 {
			ctx.Log().WithField("allocation-id", req.AllocationID).Error(
				"set max_slots_per_pod > 0 to schedule tasks with slots")
			return
		}

		if req.SlotsNeeded <= k.config.MaxSlotsPerPod {
			numPods = 1
			slotsPerPod = req.SlotsNeeded
		} else {
			if req.SlotsNeeded%k.config.MaxSlotsPerPod != 0 {
				ctx.Log().WithField("allocation-id", req.AllocationID).Errorf(
					"task number of slots (%d) is not schedulable on the configured "+
						"max_slots_per_pod (%d)", req.SlotsNeeded, k.config.MaxSlotsPerPod)
				return
			}

			numPods = req.SlotsNeeded / k.config.MaxSlotsPerPod
			slotsPerPod = k.config.MaxSlotsPerPod
		}
	}

	k.slotsUsedPerGroup[k.groups[req.Group]] += req.SlotsNeeded

	allocations := make([]sproto.Resources, 0, numPods)
	for pod := 0; pod < numPods; pod++ {
		containerID := cproto.NewID()
		allocations = append(allocations, &k8sPodResources{
			req:         req,
			podsActor:   k.podsActor,
			containerID: containerID,
			slots:       slotsPerPod,
			group:       k.groups[req.Group],
		})

		k.addrToContainerID[req.TaskActor] = containerID
		ctx.Tell(k.podsActor, kubernetes.SetPodOrder{
			QPosition: -1,
			PodID:     containerID,
		})

		k.addrToContainerID[req.TaskActor] = containerID
		k.containerIDtoAddr[containerID.String()] = req.TaskActor
	}

	assigned := sproto.ResourcesAllocated{ID: req.AllocationID, Resources: allocations}
	k.reqList.SetAllocationsRaw(req.TaskActor, &assigned)
	req.TaskActor.System().Tell(req.TaskActor, assigned)

	ctx.Log().
		WithField("allocation-id", req.AllocationID).
		WithField("task-handler", req.TaskActor.Address()).
		Infof("resources assigned with %d pods", numPods)
}

func (k *kubernetesResourceManager) resourcesReleased(ctx *actor.Context, handler *actor.Ref) {
	ctx.Log().Infof("resources are released for %s", handler.Address())
	k.reqList.RemoveTaskByHandler(handler)

	deleteID := ""
	for id, addr := range k.containerIDtoAddr {
		if addr == handler {
			deleteID = id
			delete(k.containerIDtoAddr, deleteID)
			break
		}
	}

	if req, ok := k.reqList.GetTaskByHandler(handler); ok {
		group := k.groups[handler]

		if group != nil {
			k.slotsUsedPerGroup[group] -= req.SlotsNeeded
		}
	}
}

func (k *kubernetesResourceManager) getOrCreateGroup(
	ctx *actor.Context,
	handler *actor.Ref,
) *group {
	if g, ok := k.groups[handler]; ok {
		return g
	}
	priority := config.KubernetesDefaultPriority
	g := &group{handler: handler, weight: 1, priority: &priority}

	k.groups[handler] = g
	k.slotsUsedPerGroup[g] = 0

	if ctx != nil && handler != nil { // ctx is nil only for testing purposes.
		actors.NotifyOnStop(ctx, handler, groupActorStopped{})
	}
	return g
}

func (k *kubernetesResourceManager) schedulePendingTasks(ctx *actor.Context) {
	for it := k.reqList.iterator(); it.next(); {
		req := it.value()
		group := k.groups[req.Group]
		assigned := k.reqList.GetAllocations(req.TaskActor)
		if !assignmentIsScheduled(assigned) {
			if maxSlots := group.maxSlots; maxSlots != nil {
				if k.slotsUsedPerGroup[group]+req.SlotsNeeded > *maxSlots {
					continue
				}
			}

			k.assignResources(ctx, req)
		}
	}
}

type k8sPodResources struct {
	req         *sproto.AllocateRequest
	podsActor   *actor.Ref
	group       *group
	containerID cproto.ID
	slots       int
}

// Summary summarizes a container allocation.
func (p k8sPodResources) Summary() sproto.ResourcesSummary {
	return sproto.ResourcesSummary{
		AllocationID:  p.req.AllocationID,
		ResourcesID:   sproto.ResourcesID(p.containerID),
		ResourcesType: sproto.ResourcesTypeK8sPod,
		AgentDevices: map[aproto.ID][]device.Device{
			// TODO: Make it more obvious k8s can't be trusted.
			aproto.ID(p.podsActor.Address().Local()): nil,
		},

		ContainerID: &p.containerID,
	}
}

// Start notifies the pods actor that it should launch a pod for the provided task spec.
func (p k8sPodResources) Start(
	ctx *actor.Context, logCtx logger.Context, spec tasks.TaskSpec, rri sproto.ResourcesRuntimeInfo,
) {
	spec.ContainerID = string(p.containerID)
	spec.ResourcesID = string(p.containerID)
	spec.AllocationID = string(p.req.AllocationID)
	spec.AllocationSessionToken = rri.Token
	spec.TaskID = string(p.req.TaskID)
	spec.UseHostMode = rri.IsMultiAgent
	spec.ResourcesConfig.SetPriority(p.group.priority)
	if spec.LoggingFields == nil {
		spec.LoggingFields = map[string]string{}
	}
	spec.LoggingFields["allocation_id"] = spec.AllocationID
	spec.LoggingFields["task_id"] = spec.TaskID
	spec.ExtraEnvVars[sproto.ResourcesTypeEnvVar] = string(sproto.ResourcesTypeK8sPod)
	ctx.Tell(p.podsActor, kubernetes.StartTaskPod{
		TaskActor:  p.req.TaskActor,
		Spec:       spec,
		Slots:      p.slots,
		Rank:       rri.AgentRank,
		LogContext: logCtx,
	})
}

// Kill notifies the pods actor that it should stop the pod.
func (p k8sPodResources) Kill(ctx *actor.Context, _ logger.Context) {
	ctx.Tell(p.podsActor, kubernetes.KillTaskPod{
		PodID: p.containerID,
	})
}
