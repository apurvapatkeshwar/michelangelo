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

package oidc

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/michelangelo-ai/michelangelo/go/auth"
)

// TestAuthenticatorSatisfiesAuthFromTokenAuthenticator proves the actual
// composition this stack relies on: a real OIDC Authenticator, verifying a
// real signed token against a real (test) issuer, adapted end to end into
// an auth.Principal via auth.FromTokenAuthenticator -- not just each piece
// tested in isolation.
func TestAuthenticatorSatisfiesAuthFromTokenAuthenticator(t *testing.T) {
	issuer := newTestIssuer(t)
	oidcAuthenticator := issuer.newAuthenticator(t, Config{})
	token := issuer.signToken(t, issuer.baseClaims())

	authenticator := auth.FromTokenAuthenticator(oidcAuthenticator)
	principal, err := authenticator.Authenticate(context.Background(), token)
	require.NoError(t, err)

	assert.Equal(t, auth.PrincipalTypeUser, principal.Type)
	assert.Equal(t, "alice@example.com", principal.Name)
	assert.Equal(t, "dev,ops", principal.Attrs["groups"])
}

// TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorRejectsExpired proves
// the adapter surfaces the OIDC authenticator's own rejection (not just its
// successes) as an Authenticate error.
func TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorRejectsExpired(t *testing.T) {
	issuer := newTestIssuer(t)
	oidcAuthenticator := issuer.newAuthenticator(t, Config{})
	claims := issuer.baseClaims()
	claims["exp"] = 1 // long expired
	token := issuer.signToken(t, claims)

	authenticator := auth.FromTokenAuthenticator(oidcAuthenticator)
	_, err := authenticator.Authenticate(context.Background(), token)
	assert.Error(t, err)
}
