package internal

import (
	//	"fmt"
	"net"
	"os"

	"github.com/hashicorp/hcl/v2/hclsimple"
	"github.com/pkg/errors"
	log "github.com/sirupsen/logrus"

	"github.com/determined-ai/determined/master/pkg/device"
	"github.com/determined-ai/determined/master/pkg/vipu/gc"
)

const VIPU_CLI_CONFIG_PATH = "/etc/vipu/vipu-cli.hcl"

type VIPUCliConfig struct {
        APIHost string `hcl:"api-host"`
        APIPort string `hcl:"api-port,optional"`
}

func detectIPUs() ([]device.Device, error) {

	log.Infof("Detecting IPU POD")

	if _, err := os.Stat(VIPU_CLI_CONFIG_PATH); errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}

        var vipuCliCfg VIPUCliConfig

	err := hclsimple.DecodeFile(VIPU_CLI_CONFIG_PATH, nil, &vipuCliCfg)
	if err != nil {
		log.WithError(err).Warnf(
			"error while parsing V-IPU client config file")
		return nil, nil
	}
	log.Infof("VIPU cfg %+v", vipuCliCfg)

	port := "8090"
	if vipuCliCfg.APIPort != "" {
		port = vipuCliCfg.APIPort
	}
	addr := net.JoinHostPort(vipuCliCfg.APIHost, port)

	vipuCli := gc.NewVipuClient(addr)
	parts, err := vipuCli.GetAvailableNumPartitions(1)
	defer vipuCli.Close()
	if err != nil {
		log.WithError(err).Warnf(
			"error while getting partitions from V-IPU server")
		return nil, nil
	}

	log.Infof("VIPU parts %+v", parts)

	result := []device.Device{}
	result = append(result, device.Device{
		ID:    device.ID(0),
		Brand: "Graphcore IPU POD",
		UUID:  addr,
		Type:  device.VPOD,
	})

	return result, nil
}
