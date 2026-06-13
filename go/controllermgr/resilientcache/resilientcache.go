package resilientcache

import (
	"context"
	"fmt"
	"sync"
	"time"

	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/cache"
)

const syncTimeout = 2 * time.Minute

type resilientCache struct {
	cache.Cache

	once       sync.Once
	timedOut   bool
}

func (c *resilientCache) WaitForCacheSync(ctx context.Context) bool {
	if c.timedOut {
		return true
	}

	syncCtx, cancel := context.WithTimeout(ctx, syncTimeout)
	defer cancel()

	if c.Cache.WaitForCacheSync(syncCtx) {
		return true
	}

	c.once.Do(func() {
		c.timedOut = true
		fmt.Printf("WARNING: cache sync timed out after %s — some informers failed to sync, proceeding in degraded mode\n", syncTimeout)
	})
	return true
}

// NewCacheFunc returns a cache.NewCacheFunc that wraps the default cache
// with a sync timeout so one bad CR cannot block the entire controller manager.
func NewCacheFunc() cache.NewCacheFunc {
	return func(config *rest.Config, opts cache.Options) (cache.Cache, error) {
		c, err := cache.New(config, opts)
		if err != nil {
			return nil, err
		}
		return &resilientCache{Cache: c}, nil
	}
}
