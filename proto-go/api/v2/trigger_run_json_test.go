package v2

import (
	"strings"
	"testing"

	"github.com/gogo/protobuf/jsonpb"
	"github.com/michelangelo-ai/michelangelo/go/kubeproto/util"
)

// TestUpdateTriggerRunUnmarshalRFC3339Timestamp reproduces the server-side bug
// where the API returns 400 on UpdateTriggerRun when the request includes an
// RFC3339 creationTimestamp (the format connect-es/@bufbuild/protobuf uses for
// google.protobuf.Timestamp / metav1.Time).
//
// gogo's jsonpb v1.3.2 dispatches to JSONPBUnmarshaler (UnmarshalJSONPB) on nested
// messages, not json.Unmarshaler. Without UnmarshalJSONPB on TriggerRun, gogo tries
// to decode creationTimestamp itself and fails: it expects a JSON object but receives
// an RFC3339 string. The fix adds UnmarshalJSONPB, which pre-processes int64 fields
// and then delegates to encoding/json (which handles metav1.Time.UnmarshalJSON).
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
