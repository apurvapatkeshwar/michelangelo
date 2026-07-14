package triggerrun

// Manual repro for the STG incident where the notification-drift fix (actual_notifications,
// see cron_trigger.go's notifDrifted) caused unconditional retries against Temporal schedules
// that had already exhausted history.maximumSignalsPerExecution, producing
// "exceeded workflow execution limit for signal events" and an 18 calls/sec storm for 3 triggers.
//
// Not run in CI. To run against a local Temporal dev server:
//
//	temporal server start-dev --port 17233 --ui-port 18233 \
//	    --dynamic-config-value history.maximumSignalsPerExecution=5 --headless
//	MA_TRIGGER_REPRO=1 go test ./components/triggerrun/... -run TestSignalStormRepro -v -timeout 120s

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	temporalclientpkg "github.com/michelangelo-ai/michelangelo/go/base/workflowclient/temporalclient"
	clientInterface "github.com/michelangelo-ai/michelangelo/go/base/workflowclient/interface"
	temporalClient "go.temporal.io/sdk/client"
)

func TestSignalStormRepro(t *testing.T) {
	if os.Getenv("MA_TRIGGER_REPRO") == "" {
		t.Skip("manual repro only; set MA_TRIGGER_REPRO=1 to run against a local Temporal dev server")
	}

	ctx := context.Background()
	sdkClient, err := temporalClient.Dial(temporalClient.Options{
		HostPort:  "localhost:17233",
		Namespace: "default",
	})
	if err != nil {
		t.Fatalf("dial temporal: %v", err)
	}
	defer sdkClient.Close()

	tc := &temporalclientpkg.TemporalClient{
		Client:   sdkClient,
		Provider: "temporal",
		Domain:   "default",
	}

	wid := "repro-trigger-1"

	// Step 1: create the schedule exactly the way cronTrigger.Run() does.
	_, err = tc.StartWorkflow(ctx, clientInterface.StartWorkflowOptions{
		ID:                              wid,
		TaskList:                        "trigger_run",
		ExecutionStartToCloseTimeout:    time.Hour * 24 * 365,
		DecisionTaskStartToCloseTimeout: 30 * time.Second,
		CronSchedule:                    "* * * * *",
	}, "trigger.CronTrigger", "initial-args")
	if err != nil {
		t.Fatalf("create schedule: %v", err)
	}
	t.Logf("created schedule for workflow id %q", wid)

	// Step 2: simulate the reconcile hot-loop. notifDrifted is stuck true forever
	// because Status.ActualNotifications is only persisted on a *successful*
	// UpdateTrigger call (cron_trigger.go:285-287) -- so every iteration below
	// re-sends the same "drifted" update, exactly like a real reconcile storm.
	var (
		calls, failures int
		firstLimitErrAt int
		start           = time.Now()
	)
	deadline := start.Add(30 * time.Second)
	for time.Now().Before(deadline) {
		calls++
		err := tc.UpdateTrigger(ctx, wid, "", nil, []interface{}{"drifted-args"})
		if err != nil {
			failures++
			if strings.Contains(err.Error(), "exceeded workflow execution limit for signal events") && firstLimitErrAt == 0 {
				firstLimitErrAt = calls
				t.Logf("call #%d: reproduced target error: %v", calls, err)
			}
		}
	}
	elapsed := time.Since(start)
	rps := float64(calls) / elapsed.Seconds()

	t.Logf("total calls=%d failures=%d elapsed=%s rps=%.1f firstLimitErrorAtCall=%d",
		calls, failures, elapsed, rps, firstLimitErrAt)

	if firstLimitErrAt == 0 {
		t.Fatalf("did not reproduce 'exceeded workflow execution limit for signal events' within %d calls", calls)
	}
}

// TestExhaustScheduleForLiveVerification creates (or reuses) a Temporal schedule
// for an arbitrary workflow ID and drives it past history.maximumSignalsPerExecution,
// so a real TriggerRun object can be pointed at an already-exhausted schedule for
// live controllermgr verification. Set MA_TRIGGER_EXHAUST_WID to the workflow ID
// (namespace.name of the TriggerRun you'll create).
func TestExhaustScheduleForLiveVerification(t *testing.T) {
	wid := os.Getenv("MA_TRIGGER_EXHAUST_WID")
	if wid == "" {
		t.Skip("manual repro only; set MA_TRIGGER_EXHAUST_WID=<namespace>.<name> to run")
	}

	ctx := context.Background()
	sdkClient, err := temporalClient.Dial(temporalClient.Options{
		HostPort:  "localhost:17233",
		Namespace: "default",
	})
	if err != nil {
		t.Fatalf("dial temporal: %v", err)
	}
	defer sdkClient.Close()

	tc := &temporalclientpkg.TemporalClient{
		Client:   sdkClient,
		Provider: "temporal",
		Domain:   "default",
	}

	_, err = tc.StartWorkflow(ctx, clientInterface.StartWorkflowOptions{
		ID:                              wid,
		TaskList:                        "trigger_run",
		ExecutionStartToCloseTimeout:    time.Hour * 24 * 365,
		DecisionTaskStartToCloseTimeout: 30 * time.Second,
		CronSchedule:                    "* * * * *",
	}, "trigger.CronTrigger", "initial-args")
	if err != nil {
		t.Fatalf("create schedule: %v", err)
	}
	t.Logf("created schedule for workflow id %q", wid)

	for i := 0; i < 20; i++ {
		err := tc.UpdateTrigger(ctx, wid, "", nil, []interface{}{"drifted-args"})
		if err != nil && strings.Contains(err.Error(), "exceeded workflow execution limit for signal events") {
			t.Logf("schedule for %q is now exhausted after %d calls: %v", wid, i+1, err)
			return
		}
	}
	t.Fatalf("failed to exhaust schedule for %q within 20 calls", wid)
}
