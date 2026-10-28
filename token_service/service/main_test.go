package service

import (
	"os"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

var testService *Service

// Create a service which can be fetched from any of the tests
func TestMain(m *testing.M) {

	gin.SetMode(gin.TestMode)
	os.Setenv("APP_ID", strings.Repeat("a", 32))
	os.Setenv("APP_CERTIFICATE", strings.Repeat("b", 32))
	os.Setenv("SERVER_PORT", "8080")
	testService = NewService()
	os.Exit(m.Run())
}
