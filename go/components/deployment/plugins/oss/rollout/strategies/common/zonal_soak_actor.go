package common

import (
	"context"
	"fmt"
	"time"

	"github.com/gogo/protobuf/types"
	"go.uber.org/zap"

	conditionInterfaces "github.com/michelangelo-ai/michelangelo/go/base/conditions/interfaces"
	conditionsutil "github.com/michelangelo-ai/michelangelo/go/base/conditions/utils"
	osscommon "github.com/michelangelo-ai/michelangelo/go/components/deployment/plugins/oss/common"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

var _ conditionInterfaces.ConditionActor[*v2pb.Deployment] = &ZonalSoakActor{}

// ZonalSoakActor enforces a soak period after a cluster's model is live before
// the engine advances to the next cluster. It mirrors the internal zonal actor's
// soak: once the model is confirmed loaded, it records the timestamp in the
// condition Metadata (as a DoubleValue unix seconds) and returns FALSE until
// rolloutPeriodInSeconds have elapsed.
type ZonalSoakActor struct {
	target                 *v2pb.ClusterTarget
	rolloutPeriodInSeconds float64
	logger                 *zap.Logger
}

// NewZonalSoakActor creates a soak gate for the given cluster.
func NewZonalSoakActor(target *v2pb.ClusterTarget, rolloutPeriodInSeconds float64, logger *zap.Logger) *ZonalSoakActor {
	return &ZonalSoakActor{
		target:                 target,
		rolloutPeriodInSeconds: rolloutPeriodInSeconds,
		logger:                 logger,
	}
}

// GetType returns a unique condition key per cluster.
func (a *ZonalSoakActor) GetType() string {
	return osscommon.ActorTypeZonalRollout + "-soak-" + a.target.GetClusterId()
}

// Retrieve checks whether the soak period has elapsed. Mirrors the internal
// zonal actor: reads startedAt from Metadata, compares to now.
func (a *ZonalSoakActor) Retrieve(ctx context.Context, deployment *v2pb.Deployment, condition *apipb.Condition) (*apipb.Condition, error) {
	// Already confirmed done in a prior reconcile.
	if condition.Status == apipb.CONDITION_STATUS_TRUE {
		return conditionsutil.GenerateTrueCondition(condition), nil
	}

	// No soak required — pass immediately.
	if a.rolloutPeriodInSeconds <= 0 {
		return conditionsutil.GenerateTrueCondition(condition), nil
	}

	startedAt, ok := a.readStartTime(condition)
	if !ok {
		// Run hasn't fired yet.
		return conditionsutil.GenerateFalseCondition(condition, "ZonalSoakPending",
			fmt.Sprintf("cluster %s: waiting for soak to begin", a.target.GetClusterId())), nil
	}

	elapsed := time.Since(time.Unix(int64(startedAt), 0)).Seconds()
	remaining := a.rolloutPeriodInSeconds - elapsed
	if remaining > 0 {
		a.logger.Info("zonal soak in progress",
			zap.String("cluster", a.target.GetClusterId()),
			zap.Float64("elapsedSeconds", elapsed),
			zap.Float64("totalSeconds", a.rolloutPeriodInSeconds))
		return conditionsutil.GenerateFalseCondition(condition, "ZonalSoaking",
			fmt.Sprintf("cluster %s: soaking %.0fs / %.0fs", a.target.GetClusterId(), elapsed, a.rolloutPeriodInSeconds)), nil
	}

	a.logger.Info("zonal soak complete, advancing to next cluster",
		zap.String("cluster", a.target.GetClusterId()),
		zap.Float64("soakSeconds", a.rolloutPeriodInSeconds))
	return conditionsutil.GenerateTrueCondition(condition), nil
}

// Run records the soak start time in condition Metadata as a DoubleValue.
func (a *ZonalSoakActor) Run(ctx context.Context, deployment *v2pb.Deployment, condition *apipb.Condition) (*apipb.Condition, error) {
	if a.rolloutPeriodInSeconds <= 0 {
		return conditionsutil.GenerateTrueCondition(condition), nil
	}

	if err := a.writeStartTime(condition, float64(time.Now().Unix())); err != nil {
		return conditionsutil.GenerateFalseCondition(condition, "ZonalSoakMetaError", err.Error()), nil
	}

	a.logger.Info("zonal soak started",
		zap.String("cluster", a.target.GetClusterId()),
		zap.Float64("soakSeconds", a.rolloutPeriodInSeconds))
	return conditionsutil.GenerateFalseCondition(condition, "ZonalSoakStarted",
		fmt.Sprintf("cluster %s: soak started, waiting %.0fs", a.target.GetClusterId(), a.rolloutPeriodInSeconds)), nil
}

func (a *ZonalSoakActor) readStartTime(condition *apipb.Condition) (float64, bool) {
	if condition.Metadata == nil {
		return 0, false
	}
	val := &types.DoubleValue{}
	if err := types.UnmarshalAny(condition.Metadata, val); err != nil {
		return 0, false
	}
	return val.Value, val.Value > 0
}

func (a *ZonalSoakActor) writeStartTime(condition *apipb.Condition, unixSeconds float64) error {
	metadata, err := types.MarshalAny(&types.DoubleValue{Value: unixSeconds})
	if err != nil {
		return err
	}
	condition.Metadata = metadata
	return nil
}
