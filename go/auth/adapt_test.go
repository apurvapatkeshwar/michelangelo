// Copyright (c) 2023 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package auth

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

type stubTokenAuthenticator struct {
	response *authenticator.Response
	ok       bool
	err      error
	gotToken string
}

func (s *stubTokenAuthenticator) AuthenticateToken(_ context.Context, token string) (*authenticator.Response, bool, error) {
	s.gotToken = token
	return s.response, s.ok, s.err
}

type stubAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
	got      authorizer.Attributes
}

func (s *stubAuthorizer) Authorize(_ context.Context, attributes authorizer.Attributes) (authorizer.Decision, string, error) {
	s.got = attributes
	return s.decision, s.reason, s.err
}

var alice = &user.DefaultInfo{Name: "alice@example.com", UID: "uid-1", Groups: []string{"dev", "platform"}}

// --- FromTokenAuthenticator ---

func TestFromTokenAuthenticatorMapsUserInfoToPrincipal(t *testing.T) {
	stub := &stubTokenAuthenticator{response: &authenticator.Response{User: alice}, ok: true}
	principal, err := FromTokenAuthenticator(stub).Authenticate(context.Background(), "tok-123")
	require.NoError(t, err)
	assert.Equal(t, "tok-123", stub.gotToken)
	assert.Equal(t, PrincipalTypeUser, principal.Type)
	assert.Equal(t, "alice@example.com", principal.Name)
	assert.Equal(t, "dev,platform", principal.Attrs[groupsAttr])
	assert.Equal(t, "uid-1", principal.Attrs[uidAttr])
}

func TestFromTokenAuthenticatorOmitsEmptyAttrs(t *testing.T) {
	bareUser := &user.DefaultInfo{Name: "svc"}
	stub := &stubTokenAuthenticator{response: &authenticator.Response{User: bareUser}, ok: true}
	principal, err := FromTokenAuthenticator(stub).Authenticate(context.Background(), "tok")
	require.NoError(t, err)
	assert.Empty(t, principal.Attrs[groupsAttr])
	assert.Empty(t, principal.Attrs[uidAttr])
	assert.NotContains(t, principal.Attrs, groupsAttr)
	assert.NotContains(t, principal.Attrs, uidAttr)
}

func TestFromTokenAuthenticatorPropagatesError(t *testing.T) {
	stub := &stubTokenAuthenticator{err: errors.New("bad signature")}
	_, err := FromTokenAuthenticator(stub).Authenticate(context.Background(), "tok")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "bad signature")
}

func TestFromTokenAuthenticatorFailsClosedWhenNotOK(t *testing.T) {
	stub := &stubTokenAuthenticator{ok: false}
	_, err := FromTokenAuthenticator(stub).Authenticate(context.Background(), "tok")
	require.Error(t, err)
}

func TestFromTokenAuthenticatorFailsClosedOnNilResponse(t *testing.T) {
	stub := &stubTokenAuthenticator{ok: true, response: nil}
	_, err := FromTokenAuthenticator(stub).Authenticate(context.Background(), "tok")
	require.Error(t, err)
}

// --- K8sRBACAuth.UserAuthenticated ---

func TestK8sRBACAuthUserAuthenticatedRequiresPrincipal(t *testing.T) {
	a := NewK8sRBACAuth(&stubAuthorizer{})
	ok, err := a.UserAuthenticated(context.Background())
	assert.False(t, ok)
	require.Error(t, err)
}

func TestK8sRBACAuthUserAuthenticatedTrueWithPrincipal(t *testing.T) {
	a := NewK8sRBACAuth(&stubAuthorizer{})
	ctx := WithPrincipal(context.Background(), &Principal{Type: PrincipalTypeUser, Name: "alice"})
	ok, err := a.UserAuthenticated(ctx)
	assert.True(t, ok)
	assert.NoError(t, err)
}

// --- K8sRBACAuth.UserAuthorized ---

func principalCtx() context.Context {
	return WithPrincipal(context.Background(), &Principal{
		Type: PrincipalTypeUser,
		Name: "alice@example.com",
		Attrs: map[string]string{
			groupsAttr: "dev,platform",
			uidAttr:    "uid-1",
		},
	})
}

func TestK8sRBACAuthUserAuthorizedBuildsAttributesAndAllows(t *testing.T) {
	stub := &stubAuthorizer{decision: authorizer.DecisionAllow}
	a := NewK8sRBACAuth(stub)
	ok, err := a.UserAuthorized(principalCtx(), "team-a", Create, "Cluster")
	require.NoError(t, err)
	assert.True(t, ok)
	assert.Equal(t, "alice@example.com", stub.got.GetUser().GetName())
	assert.Equal(t, []string{"dev", "platform"}, stub.got.GetUser().GetGroups())
	assert.Equal(t, "create", stub.got.GetVerb())
	assert.Equal(t, APIGroup, stub.got.GetAPIGroup())
	assert.Equal(t, "clusters", stub.got.GetResource())
	assert.Equal(t, "team-a", stub.got.GetNamespace())
	assert.True(t, stub.got.IsResourceRequest())
}

func TestK8sRBACAuthUserAuthorizedDeniesUnlessAllow(t *testing.T) {
	for _, decision := range []authorizer.Decision{authorizer.DecisionDeny, authorizer.DecisionNoOpinion} {
		stub := &stubAuthorizer{decision: decision, reason: "no RoleBinding"}
		a := NewK8sRBACAuth(stub)
		ok, err := a.UserAuthorized(principalCtx(), "team-a", Delete, "Model")
		assert.False(t, ok)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no RoleBinding")
	}
}

func TestK8sRBACAuthUserAuthorizedFailsClosed(t *testing.T) {
	stub := &stubAuthorizer{err: errors.New("apiserver unreachable")}
	a := NewK8sRBACAuth(stub)

	ok, err := a.UserAuthorized(principalCtx(), "team-a", Get, "Model")
	assert.False(t, ok)
	require.Error(t, err)

	ok, err = a.UserAuthorized(principalCtx(), "team-a", Action("Patch"), "Model")
	assert.False(t, ok)
	require.Error(t, err)

	ok, err = a.UserAuthorized(principalCtx(), "team-a", Get, "NotAKind")
	assert.False(t, ok)
	require.Error(t, err)

	ok, err = a.UserAuthorized(context.Background(), "team-a", Get, "Model")
	assert.False(t, ok)
	require.Error(t, err)
}
