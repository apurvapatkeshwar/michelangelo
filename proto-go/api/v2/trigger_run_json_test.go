package v2

import (
	"strings"
	"testing"

	"github.com/gogo/protobuf/jsonpb"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/util"
)

// TestUpdateTriggerRunUnmarshalRFC3339Timestamp reproduces the server-side bug
// where the API cannot unmarshal a TriggerRun containing an RFC3339 creationTimestamp.
//
// The server uses gogo's jsonpb to decode incoming JSON requests. When jsonpb
// processes UpdateTriggerRunRequest, it delegates the "triggerRun" field to
// TriggerRun's custom UnmarshalJSON (if present). Without that method, jsonpb
// tries to decode the field itself and fails: it expects a JSON object for
// metav1.Time (which is embedded in ObjectMeta) but receives an RFC3339 string.
//
// The fix adds UnmarshalJSON to TriggerRun, which catches that error and falls
// back to encoding/json — which correctly handles metav1.Time via its own
// UnmarshalJSON implementation.
func TestUpdateTriggerRunUnmarshalRFC3339Timestamp(t *testing.T) {
	// This is the JSON the UI sends when calling UpdateTriggerRun —
	// a TriggerRun read from the API and sent back verbatim, including
	// creationTimestamp as an RFC3339 string.
	body := `{
		"triggerRun": {
			"metadata": {
				"name": "test-trigger",
				"namespace": "ma-dev-test",
				"creationTimestamp": "2026-06-02T21:14:52Z"
			},
			"spec": {
				"trigger": {
					"cronSchedule": {"cron": "0 0 * * *"}
				},
				"action": "TRIGGER_RUN_ACTION_KILL"
			}
		}
	}`

	req := &UpdateTriggerRunRequest{}
	err := (&jsonpb.Unmarshaler{
		AllowUnknownFields: true,
		AnyResolver:        &util.GenericResolver{},
	}).Unmarshal(strings.NewReader(body), req)

	if err != nil {
		t.Fatalf("UpdateTriggerRun failed to unmarshal RFC3339 creationTimestamp: %v", err)
	}

	if req.TriggerRun.Name != "test-trigger" {
		t.Errorf("name: got %q, want %q", req.TriggerRun.Name, "test-trigger")
	}
	if req.TriggerRun.CreationTimestamp.IsZero() {
		t.Error("creationTimestamp was lost during unmarshal")
	}
}
