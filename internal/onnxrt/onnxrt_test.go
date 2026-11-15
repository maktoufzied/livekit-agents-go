package onnxrt

import "testing"

func TestCandidates(t *testing.T) {
	t.Setenv(LibraryEnv, "/from/env/libonnxruntime.so")
	if got := Candidates("/explicit.so"); len(got) != 1 || got[0] != "/explicit.so" {
		t.Fatalf("explicit path: %v", got)
	}
	if got := Candidates(""); len(got) != 1 || got[0] != "/from/env/libonnxruntime.so" {
		t.Fatalf("environment: %v", got)
	}
	t.Setenv(LibraryEnv, "")
	if got := Candidates(""); len(got) == 0 {
		t.Fatal("no default candidates")
	}
}
