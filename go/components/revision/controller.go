package revision

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/michelangelo-ai/michelangelo/go/api"
	apiHandler "github.com/michelangelo-ai/michelangelo/go/api/handler"
	apiutils "github.com/michelangelo-ai/michelangelo/go/api/utils"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
	"go.uber.org/zap"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

const _requestTimeoutSec = 30

// Reconciler watches Revision CRs and dispatches to the Handler registered
// for each revision's Spec.BaseType.
type Reconciler struct {
	api.Handler
	apiHandlerFactory apiHandler.Factory
	logger            *zap.Logger
	handlers          map[metav1.TypeMeta]Handler
}

// NewReconciler constructs a Reconciler with the given handlers.
func NewReconciler(
	apiHandlerFactory apiHandler.Factory,
	logger *zap.Logger,
	handlers []Handler,
) *Reconciler {
	m := make(map[metav1.TypeMeta]Handler, len(handlers))
	for _, h := range handlers {
		m[h.TypeMeta()] = h
	}
	return &Reconciler{
		apiHandlerFactory: apiHandlerFactory,
		logger:            logger.With(zap.String("controller", "revision")),
		handlers:          m,
	}
}

// Reconcile implements reconcile.Reconciler. Status is persisted when the
// handler mutates any field in rev.Status, even if the handler returns an error.
func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	ctx, cancel := context.WithTimeout(ctx, _requestTimeoutSec*time.Second)
	defer cancel()

	logger := r.logger.With(zap.String("namespace-name", req.NamespacedName.String()))
	logger.Debug("Reconcile called",
		zap.String("namespace", req.Namespace),
		zap.String("name", req.Name),
	)

	rev := &v2pb.Revision{}
	if err := r.Get(ctx, req.Namespace, req.Name, &metav1.GetOptions{}, rev); err != nil {
		if apiutils.IsNotFoundError(err) {
			logger.Debug("revision not found; ignoring")
			return ctrl.Result{}, nil
		}
		logger.Debug("Get revision failed", zap.Error(err))
		return ctrl.Result{}, err
	}
	logger.Debug("Get revision succeeded",
		zap.String("state", rev.Status.GetState().String()),
		zap.Bool("hasBaseType", rev.Spec.BaseType != nil),
	)

	if !rev.GetDeletionTimestamp().IsZero() {
		logger.Debug("revision is being deleted; skipping reconcile")
		return ctrl.Result{}, nil
	}

	if apiutils.IsImmutable(rev) {
		logger.Debug("revision is immutable; skipping reconcile")
		return ctrl.Result{}, nil
	}

	if rev.Status.State == v2pb.REVISION_STATE_READY || rev.Status.State == v2pb.REVISION_STATE_ERROR {
		logger.Info("revision in terminal state; marking immutable",
			zap.String("state", rev.Status.State.String()))
		apiutils.MarkImmutable(rev)
		if err := r.Update(ctx, rev, &metav1.UpdateOptions{}); err != nil {
			return ctrl.Result{}, fmt.Errorf("mark revision immutable %s/%s: %w", req.Namespace, req.Name, err)
		}
		return ctrl.Result{}, nil
	}

	if rev.Spec.BaseType == nil {
		logger.Debug("revision has no BaseType; skipping reconcile")
		return ctrl.Result{}, nil
	}

	key := metav1.TypeMeta{
		APIVersion: rev.Spec.BaseType.APIVersion,
		Kind:       rev.Spec.BaseType.Kind,
	}
	h, ok := r.handlers[key]
	if !ok {
		logger.Debug("no handler registered for BaseType; skipping reconcile",
			zap.String("apiVersion", key.APIVersion),
			zap.String("kind", key.Kind),
			zap.Int("registeredHandlers", len(r.handlers)),
		)
		return ctrl.Result{}, nil
	}

	logger.Debug("dispatching to handler",
		zap.String("apiVersion", key.APIVersion),
		zap.String("kind", key.Kind),
	)

	original := rev.DeepCopy()

	result, handlerErr := h.Reconcile(ctx, rev)
	if handlerErr != nil {
		logger.Debug("handler reconcile failed",
			zap.String("apiVersion", key.APIVersion),
			zap.String("kind", key.Kind),
			zap.Error(handlerErr),
		)
		handlerErr = fmt.Errorf("handler reconcile for %s/%s: %w", key.APIVersion, key.Kind, handlerErr)
	} else {
		logger.Debug("handler reconcile succeeded",
			zap.String("newState", rev.Status.GetState().String()),
			zap.Bool("requeue", result.Requeue),
			zap.Duration("requeueAfter", result.RequeueAfter),
		)
	}

	var updateErr error
	statusChanged := !reflect.DeepEqual(original.Status, rev.Status)
	logger.Debug("status diff check",
		zap.Bool("statusChanged", statusChanged),
		zap.String("originalState", original.Status.GetState().String()),
		zap.String("currentState", rev.Status.GetState().String()),
	)
	if statusChanged {
		logger.Debug("status changed; persisting update")
		if err := r.UpdateStatus(ctx, rev, &metav1.UpdateOptions{}); err != nil {
			logger.Debug("UpdateStatus failed", zap.Error(err))
			updateErr = fmt.Errorf("update revision status %s/%s: %w", req.Namespace, req.Name, err)
		} else {
			logger.Debug("UpdateStatus succeeded")
		}
	}

	return result, errors.Join(handlerErr, updateErr)
}

// Register sets up the Revision controller with the controller-runtime manager.
func (r *Reconciler) Register(mgr ctrl.Manager) error {
	handler, err := r.apiHandlerFactory.GetAPIHandler(mgr.GetClient())
	if err != nil {
		r.logger.Error("GetAPIHandler failed", zap.Error(err))
		return err
	}
	r.Handler = handler

	return ctrl.NewControllerManagedBy(mgr).
		For(&v2pb.Revision{}).
		WithEventFilter(predicate.NewPredicateFuncs(func(object client.Object) bool {
			rev, ok := object.(*v2pb.Revision)
			if !ok {
				r.logger.Debug("event filter: object is not a Revision",
					zap.String("type", fmt.Sprintf("%T", object)))
				return false
			}
			if rev.Spec.BaseType == nil {
				r.logger.Debug("event filter: revision has no BaseType",
					zap.String("name", rev.GetName()),
					zap.String("namespace", rev.GetNamespace()))
				return false
			}
			key := metav1.TypeMeta{
				APIVersion: rev.Spec.BaseType.APIVersion,
				Kind:       rev.Spec.BaseType.Kind,
			}
			_, ok = r.handlers[key]
			if !ok {
				r.logger.Debug("event filter: no handler for BaseType",
					zap.String("name", rev.GetName()),
					zap.String("apiVersion", key.APIVersion),
					zap.String("kind", key.Kind))
			}
			return ok
		})).
		Complete(r)
}
