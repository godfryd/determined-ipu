// Package command provides utilities for commands.
//nolint:dupl
package command

import (
	"net/http"

	"github.com/labstack/echo/v4"

	"github.com/determined-ai/determined/master/internal/db"
	"github.com/determined-ai/determined/master/internal/task"
	"github.com/determined-ai/determined/master/pkg/actor"
	"github.com/determined-ai/determined/master/pkg/model"
	"github.com/determined-ai/determined/master/pkg/tasks"
	"github.com/determined-ai/determined/proto/pkg/apiv1"
	"github.com/determined-ai/determined/proto/pkg/commandv1"
)

type commandManager struct {
	db         *db.PgDB
	taskLogger *task.Logger
}

func (c *commandManager) Receive(ctx *actor.Context) error {
	switch msg := ctx.Message().(type) {
	case actor.PreStart, actor.PostStop, actor.ChildFailed, actor.ChildStopped:

	case *apiv1.GetCommandsRequest:
		resp := &apiv1.GetCommandsResponse{}
		users := make(map[string]bool)
		for _, user := range msg.Users {
			users[user] = true
		}
		for _, command := range ctx.AskAll(&commandv1.Command{}, ctx.Children()...).GetAll() {
			if typed := command.(*commandv1.Command); len(users) == 0 || users[typed.Username] {
				resp.Commands = append(resp.Commands, typed)
			}
		}
		ctx.Respond(resp)

	case tasks.GenericCommandSpec:
		taskID := model.NewTaskID()
		jobID := model.NewJobID()
		if err := createGenericCommandActor(
			ctx, c.db, c.taskLogger, taskID, model.TaskTypeCommand, jobID, model.JobTypeCommand, msg,
		); err != nil {
			ctx.Log().WithError(err).Error("failed to launch command")
			ctx.Respond(err)
		} else {
			ctx.Respond(taskID)
		}

	case echo.Context:
		ctx.Respond(echo.NewHTTPError(http.StatusNotFound, ErrAPIRemoved))

	default:
		return actor.ErrUnexpectedMessage(ctx)
	}
	return nil
}
