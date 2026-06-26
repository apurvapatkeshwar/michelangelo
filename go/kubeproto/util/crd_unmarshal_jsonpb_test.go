package util_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/gogo/protobuf/jsonpb"
	"github.com/stretchr/testify/require"

	"github.com/michelangelo-ai/michelangelo/go/kubeproto/util"
	testpb "github.com/michelangelo-ai/michelangelo/proto-go/test/kubeproto"
)

// CRD types embed metav1.ObjectMeta, which mixes two field encodings that no
// single JSON decoder handles:
//   - creationTimestamp is a metav1.Time, a Go struct that serializes as a JSON
//     string; gogo jsonpb reflects into it and fails with "cannot unmarshal
//     string into Go value of type map[string]json.RawMessage".
//   - generation is an int64; jsonpb and the protobuf-es web client serialize it
//     as a JSON string (proto3 JSON), which encoding/json refuses to decode.
//
// crd.tmpl now generates UnmarshalJSONPB for every CRD type. gogo jsonpb only
// dispatches to UnmarshalJSONPB (never UnmarshalJSON) for nested messages, e.g.
// the TriggerRun inside an UpdateTriggerRunRequest. The method coerces the
// integer fields and delegates to encoding/json, so both the controller-runtime
// path and the YARPC/UI path decode the same round-tripped payload.
func TestCRDUnmarshal_ObjectMeta_BothPaths(t *testing.T) {
	// A CRD as it comes back from the API and is echoed into an Update:
	// generation is a quoted int64, creationTimestamp is a quoted timestamp.
	payload := `{
		"metadata": {
			"name": "test",
			"namespace": "default",
			"generation": "2",
			"creationTimestamp": "2026-06-25T00:00:00Z"
		}
	}`

	t.Run("encoding/json (controller-runtime path)", func(t *testing.T) {
		// encoding/json needs the proto3 string-encoded integers coerced first,
		// exactly as the generated UnmarshalJSONPB does.
		coerced, err := util.CoerceObjectMetaIntegers([]byte(payload))
		require.NoError(t, err)

		var obj testpb.TestObject
		require.NoError(t, json.Unmarshal(coerced, &obj))
		require.Equal(t, "test", obj.Name)
		require.Equal(t, int64(2), obj.Generation)
		require.False(t, obj.CreationTimestamp.IsZero())
	})

	t.Run("gogo jsonpb (YARPC/UI path)", func(t *testing.T) {
		var obj testpb.TestObject
		err := (&jsonpb.Unmarshaler{AllowUnknownFields: true}).Unmarshal(
			strings.NewReader(payload), &obj,
		)
		require.NoError(t, err, "UnmarshalJSONPB should decode metav1.Time and string-encoded int64")
		require.Equal(t, "test", obj.Name)
		require.Equal(t, int64(2), obj.Generation)
		require.False(t, obj.CreationTimestamp.IsZero())
	})
}

func TestCoerceObjectMetaIntegers(t *testing.T) {
	t.Run("converts string-encoded generation to a number", func(t *testing.T) {
		out, err := util.CoerceObjectMetaIntegers([]byte(`{"metadata":{"generation":"7"}}`))
		require.NoError(t, err)
		require.JSONEq(t, `{"metadata":{"generation":7}}`, string(out))
	})

	t.Run("leaves string fields like resourceVersion untouched", func(t *testing.T) {
		in := `{"metadata":{"resourceVersion":"123","generation":"7"}}`
		out, err := util.CoerceObjectMetaIntegers([]byte(in))
		require.NoError(t, err)
		require.JSONEq(t, `{"metadata":{"resourceVersion":"123","generation":7}}`, string(out))
	})

	t.Run("is a no-op when integers are already numbers", func(t *testing.T) {
		in := `{"metadata":{"generation":7}}`
		out, err := util.CoerceObjectMetaIntegers([]byte(in))
		require.NoError(t, err)
		require.JSONEq(t, in, string(out))
	})
}
