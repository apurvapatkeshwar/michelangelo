package backends

import (
	"go.uber.org/fx"

	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/clientfactory"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

var Module = fx.Options(
	fx.Provide(NewBackendRegistry),
)

// Params holds dependencies for backend registry construction.
type Params struct {
	fx.In
	ClientFactory clientfactory.ClientFactory
}

// NewBackendRegistry creates and populates a backend registry with all supported backends.
func NewBackendRegistry(p Params) *Registry {
	registry := NewRegistry()

	registry.Register(v2pb.BACKEND_TYPE_TRITON, NewTritonBackend())

	registry.Register(v2pb.BACKEND_TYPE_KSERVE, NewKServeBackend(p.ClientFactory.GetDynamicClient))

	return registry
}
