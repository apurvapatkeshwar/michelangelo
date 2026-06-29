package strategies

import (
	"context"
	"fmt"
	"net/http"

	"go.uber.org/zap"
	"k8s.io/client-go/dynamic"
	"sigs.k8s.io/controller-runtime/pkg/client"

	conditionInterfaces "github.com/michelangelo-ai/michelangelo/go/base/conditions/interfaces"
	"github.com/michelangelo-ai/michelangelo/go/components/common/routing"
	osscommon "github.com/michelangelo-ai/michelangelo/go/components/deployment/plugins/oss/common"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/backends"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/clientfactory"
	modelconfig "github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/modelconfig"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"

	strategiesCommon "github.com/michelangelo-ai/michelangelo/go/components/deployment/plugins/oss/rollout/strategies/common"
)

// Params contains dependencies for strategy actor construction.
type Params struct {
	ClientFactory       clientfactory.ClientFactory
	RouteManager        routing.Manager
	BackendRegistry     *backends.Registry
	ModelConfigProvider modelconfig.ModelConfigProvider
	Logger              *zap.Logger

	// DynamicClient is the dynamic client for the control-plane cluster. Retained so that
	// actors operating on control-plane-only resources can access it directly.
	DynamicClient dynamic.Interface

	// Client is the controller-runtime client for the control-plane cluster.
	Client client.Client

	// HTTPClient is the HTTP client for the control-plane cluster.
	HTTPClient *http.Client
}

// GetActorsForStrategy returns the ordered actor chain for the deployment's rollout strategy.
// Each cluster gets its own RollingRolloutActor; the model is exposed via a single
// DiscoveryRoutingActor that adds the deployment's rule to the InferenceServer's discovery
// route. Cleanup actors follow at the end so old models are removed only after every cluster
// has flipped to the new model.
func GetActorsForStrategy(ctx context.Context, params Params, deployment *v2pb.Deployment) ([]conditionInterfaces.ConditionActor[*v2pb.Deployment], error) {
	strategy := getDeploymentStrategy(deployment)
	params.Logger.Info("Selected rollout strategy", zap.String("strategy", strategy), zap.String("deployment", deployment.Name))

	switch strategy {
	case "zonal":
		return getZonalActors(params, deployment)
	// TODO(#623): Implement blast, shadow, and disaggregated strategies.
	case "rolling":
		fallthrough
	default:
		return getRollingActors(params, deployment)
	}
}

// getRollingActors builds the per-cluster actor chain for the rolling strategy. The actor
// list is constructed from the cluster snapshot annotation written by PlacementPrepActor.
func getRollingActors(params Params, deployment *v2pb.Deployment) ([]conditionInterfaces.ConditionActor[*v2pb.Deployment], error) {
	targets, err := osscommon.ReadTargetClustersAnnotation(deployment)
	if err != nil {
		return nil, fmt.Errorf("read target clusters annotation: %w", err)
	}
	if len(targets) == 0 {
		// Annotation absent or no healthy clusters yet. Return an empty list so
		// per-cluster actors are omitted this reconcile; they are added once the
		// annotation is written and a healthy cluster is available.
		return nil, nil
	}

	// Per-cluster [RollingRollout, TrafficRouting] pairs come first, interleaved so cluster N
	// starts routing traffic as soon as its model is loaded. A single DiscoveryRoutingActor then
	// exposes the deployment via the control-plane discovery route. Per-cluster ModelCleanup
	// actors run at the end so old models are removed only after every cluster has flipped.
	actors := make([]conditionInterfaces.ConditionActor[*v2pb.Deployment], 0, 3*len(targets)+1)

	for _, target := range targets {
		actors = append(actors,
			strategiesCommon.NewRollingRolloutActor(params.ClientFactory, params.BackendRegistry, params.ModelConfigProvider, params.Logger, target),
			strategiesCommon.NewTrafficRoutingActor(params.ClientFactory, params.RouteManager, target),
		)
	}
	actors = append(actors, strategiesCommon.NewDiscoveryRoutingActor(params.DynamicClient, params.RouteManager))
	for _, target := range targets {
		actors = append(actors, strategiesCommon.NewModelCleanupActor(params.ClientFactory, params.BackendRegistry, params.ModelConfigProvider, params.Logger, target))
	}

	return actors, nil
}

// getZonalActors builds the per-cluster actor chain for the zonal strategy.
// Clusters are processed sequentially: each cluster must load the model, route
// traffic, and complete its soak period before the next cluster begins.
// Old models are cleaned up only after every cluster has advanced.
func getZonalActors(params Params, deployment *v2pb.Deployment) ([]conditionInterfaces.ConditionActor[*v2pb.Deployment], error) {
	targets, err := osscommon.ReadTargetClustersAnnotation(deployment)
	if err != nil {
		return nil, fmt.Errorf("read target clusters annotation: %w", err)
	}
	if len(targets) == 0 {
		return nil, nil
	}

	soakSeconds := deployment.Spec.GetStrategy().GetZonal().GetRolloutPeriodInSeconds()

	// For each cluster (zone): [RollingRollout → TrafficRouting → ZonalSoak]
	// The condition engine stops at the first non-TRUE actor, so cluster N+1
	// cannot start until cluster N's soak actor returns TRUE.
	// DiscoveryRouting and ModelCleanup follow once all zones are done.
	actors := make([]conditionInterfaces.ConditionActor[*v2pb.Deployment], 0, 4*len(targets)+1)

	for _, target := range targets {
		actors = append(actors,
			strategiesCommon.NewRollingRolloutActor(params.ClientFactory, params.BackendRegistry, params.ModelConfigProvider, params.Logger, target),
			strategiesCommon.NewTrafficRoutingActor(params.ClientFactory, params.RouteManager, target),
			strategiesCommon.NewZonalSoakActor(target, soakSeconds, params.Logger),
		)
	}
	actors = append(actors, strategiesCommon.NewDiscoveryRoutingActor(params.DynamicClient, params.RouteManager))
	for _, target := range targets {
		actors = append(actors, strategiesCommon.NewModelCleanupActor(params.ClientFactory, params.BackendRegistry, params.ModelConfigProvider, params.Logger, target))
	}

	return actors, nil
}

// getDeploymentStrategy determines the rollout strategy from deployment configuration.
func getDeploymentStrategy(deployment *v2pb.Deployment) string {
	switch deployment.Spec.GetStrategy().GetRolloutStrategy().(type) {
	case *v2pb.DeploymentStrategy_Zonal:
		return "zonal"
	case *v2pb.DeploymentStrategy_Rolling:
		return "rolling"
	default:
		return "rolling"
	}
}
