package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/AgoraIO-Community/go-tokenbuilder/chatTokenBuilder"
)

var chatUsernamePattern = regexp.MustCompile(`^[a-z0-9_.-]{1,64}$`)

// verifyUnityPlayer asks Unity Authentication to validate the player's bearer
// token. The requested ID must match the authenticated ID before we create an
// Agora account or issue a user-scoped Chat token.
func (s *Service) verifyUnityPlayer(ctx context.Context, authorization, playerID string) error {
	if s.unityProjectID == "" || authorization == "" || !strings.HasPrefix(authorization, "Bearer ") {
		return errors.New("Unity Authentication is required for Chat")
	}
	token := strings.TrimSpace(strings.TrimPrefix(authorization, "Bearer "))
	if token == "" || strings.ContainsAny(token, " \t\r\n") {
		return errors.New("invalid Unity Authentication token")
	}
	endpoint := strings.TrimRight(s.unityAuthURL, "/") + "/v1/users/" + url.PathEscape(playerID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("ProjectId", s.unityProjectID)
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("Unity Authentication request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Unity Authentication rejected player (%d)", resp.StatusCode)
	}
	var player struct {
		ID       string `json:"id"`
		Disabled bool   `json:"disabled"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&player); err != nil {
		return fmt.Errorf("invalid Unity Authentication response: %w", err)
	}
	if player.Disabled || player.ID != playerID {
		return errors.New("Unity Authentication player does not match requested ID")
	}
	return nil
}

func (s *Service) chatUsersURL() (string, error) {
	host := strings.TrimSpace(s.chatRESTHost)
	if host == "" {
		return "", errors.New("CHAT_REST_HOST is not configured")
	}
	if !strings.Contains(host, "://") {
		host = "https://" + host
	}
	parsed, err := url.Parse(host)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" ||
		parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", errors.New("CHAT_REST_HOST must be an HTTPS host")
	}
	return parsed.String() + "/app-id/" + url.PathEscape(s.appID) + "/users", nil
}

func (s *Service) chatRESTRequest(ctx context.Context, method, endpoint, appToken string, body io.Reader) (int, string, error) {
	req, err := http.NewRequestWithContext(ctx, method, endpoint, body)
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Authorization", "Bearer "+appToken)
	req.Header.Set("Accept", "application/json")
	if method == http.MethodPost {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	return resp.StatusCode, string(data), err
}

func activeChatUser(body, username string) bool {
	var result struct {
		Entities []struct {
			Username  string `json:"username"`
			Activated *bool  `json:"activated"`
		} `json:"entities"`
	}
	if json.Unmarshal([]byte(body), &result) != nil {
		return false
	}
	for _, entity := range result.Entities {
		if entity.Username == username && entity.Activated != nil && *entity.Activated {
			return true
		}
	}
	return false
}

// ensureChatUser is idempotent: look up first, create only when Agora reports
// a missing user, then handle a concurrent creator by querying again.
func (s *Service) ensureChatUser(ctx context.Context, username string) error {
	if !chatUsernamePattern.MatchString(username) {
		return errors.New("invalid Agora Chat username")
	}
	usersURL, err := s.chatUsersURL()
	if err != nil {
		return err
	}
	appToken, err := chatTokenBuilder.BuildChatAppToken(s.appID, s.appCertificate, 3600)
	if err != nil {
		return fmt.Errorf("cannot create Agora Chat app token: %w", err)
	}
	userURL := usersURL + "/" + url.PathEscape(username)
	status, body, err := s.chatRESTRequest(ctx, http.MethodGet, userURL, appToken, nil)
	if err != nil {
		return err
	}
	if status == http.StatusOK {
		if activeChatUser(body, username) {
			return nil
		}
		return errors.New("Agora Chat user is inactive or lookup response is invalid")
	}
	var apiError struct {
		Error string `json:"error"`
	}
	_ = json.Unmarshal([]byte(body), &apiError)
	if status != http.StatusNotFound ||
		(apiError.Error != "service_resource_not_found" && apiError.Error != "resource_not_found") {
		return fmt.Errorf("Agora Chat user lookup failed (%d): %s", status, apiError.Error)
	}

	payload, _ := json.Marshal(struct {
		Username string `json:"username"`
	}{Username: username})
	status, body, err = s.chatRESTRequest(ctx, http.MethodPost, usersURL, appToken, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	if status == http.StatusOK {
		return nil
	}
	apiError.Error = ""
	_ = json.Unmarshal([]byte(body), &apiError)
	if status == http.StatusBadRequest &&
		(apiError.Error == "duplicate_unique" || apiError.Error == "property_exists") {
		status, body, err = s.chatRESTRequest(ctx, http.MethodGet, userURL, appToken, nil)
		if err == nil && status == http.StatusOK && activeChatUser(body, username) {
			return nil
		}
	}
	return fmt.Errorf("Agora Chat user registration failed (%d): %s", status, apiError.Error)
}
