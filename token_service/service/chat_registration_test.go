package service

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatTokenRegistersBeforeIssuingToken(t *testing.T) {
	createCalls := 0
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/users/MixedCasePlayer":
			if r.Header.Get("Authorization") != "Bearer valid-token" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			w.Write([]byte(`{"id":"MixedCasePlayer","disabled":false}`))
		case strings.HasSuffix(r.URL.Path, "/users/mixedcaseplayer"):
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte(`{"error":"service_resource_not_found"}`))
		case strings.HasSuffix(r.URL.Path, "/users") && r.Method == http.MethodPost:
			createCalls++
			var payload struct {
				Username string `json:"username"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil || payload.Username != "mixedcaseplayer" {
				t.Errorf("unexpected registration body: %v, %q", err, payload.Username)
			}
			w.Write([]byte(`{"entities":[{"username":"mixedcaseplayer"}]}`))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	s := &Service{
		appID: strings.Repeat("a", 32), appCertificate: strings.Repeat("b", 32),
		chatRESTHost: server.URL, unityAuthURL: server.URL,
		unityProjectID: "project1", httpClient: server.Client(),
	}
	body := []byte(`{"tokenType":"chat","uid":"MixedCasePlayer","expire":3600}`)
	req := httptest.NewRequest(http.MethodPost, "/getToken", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer valid-token")
	response := httptest.NewRecorder()
	s.GetToken(response, req)
	if response.Code != http.StatusOK {
		t.Fatalf("expected Chat token, got %d: %s", response.Code, response.Body.String())
	}
	var result struct {
		Token string `json:"token"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || result.Token == "" {
		t.Fatalf("missing Chat token: %v", err)
	}
	if createCalls != 1 {
		t.Fatalf("expected one registration, got %d", createCalls)
	}
}

func TestEnsureChatUserCreatesOnce(t *testing.T) {
	created := false
	createCalls := 0
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "" {
			t.Error("missing app token")
		}
		switch {
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/users/player1"):
			if !created {
				w.WriteHeader(http.StatusNotFound)
				w.Write([]byte(`{"error":"service_resource_not_found"}`))
				return
			}
			w.Write([]byte(`{"entities":[{"username":"player1","activated":true}]}`))
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/users"):
			var payload struct {
				Username string `json:"username"`
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil || payload.Username != "player1" {
				t.Errorf("unexpected registration body: %v, %q", err, payload.Username)
			}
			createCalls++
			created = true
			w.Write([]byte(`{"entities":[{"username":"player1"}]}`))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	s := &Service{
		appID: strings.Repeat("a", 32), appCertificate: strings.Repeat("b", 32),
		chatRESTHost: server.URL, httpClient: server.Client(),
	}
	if err := s.ensureChatUser(context.Background(), "player1"); err != nil {
		t.Fatal(err)
	}
	if err := s.ensureChatUser(context.Background(), "player1"); err != nil {
		t.Fatal(err)
	}
	if createCalls != 1 {
		t.Fatalf("expected one registration, got %d", createCalls)
	}
}

func TestVerifyUnityPlayerChecksIdentity(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("ProjectId") != "project1" ||
			r.Header.Get("Authorization") != "Bearer valid-token" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Write([]byte(`{"id":"MixedCasePlayer","disabled":false}`))
	}))
	defer server.Close()
	s := &Service{
		unityProjectID: "project1", unityAuthURL: server.URL,
		httpClient: server.Client(),
	}
	if err := s.verifyUnityPlayer(context.Background(), "Bearer valid-token", "MixedCasePlayer"); err != nil {
		t.Fatal(err)
	}
	if err := s.verifyUnityPlayer(context.Background(), "Bearer valid-token", "AnotherPlayer"); err == nil {
		t.Fatal("accepted another player ID")
	}
	if err := s.verifyUnityPlayer(context.Background(), "Bearer invalid-token", "MixedCasePlayer"); err == nil {
		t.Fatal("accepted an invalid Unity token")
	}
	if err := s.verifyUnityPlayer(context.Background(), "", "MixedCasePlayer"); err == nil {
		t.Fatal("accepted missing Unity token")
	}
}
