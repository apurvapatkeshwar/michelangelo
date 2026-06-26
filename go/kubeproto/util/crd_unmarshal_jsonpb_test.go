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

// Regression guard for the generated CRD UnmarshalJSONPB (see crd.tmpl) and
// util.CoerceObjectMetaIntegers: a CRD round-tripped through the API must decode
// via both gogo jsonpb (the YARPC/UI path, which only ever calls UnmarshalJSONPB)
// and encoding/json (the controller-runtime path). The payload mixes the two
// fields that broke each decoder — a quoted metav1.Time and a quoted int64.
func TestCRDUnmarshal_ObjectMeta_BothPaths(t *testing.T) {
	// As returned by the API and echoed back into an Update: both fields quoted.
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
