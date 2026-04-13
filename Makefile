GO_FILES := $(shell find . -name '*.go' -not -path './.git/*' | sort)

.PHONY: fmt fmt-check smoke-sqlite-vec test test-race vet analyze

fmt:
	gofmt -w $(GO_FILES)

fmt-check:
	@test -z "$$(gofmt -l $(GO_FILES))"

smoke-sqlite-vec:
	go test ./store -run TestCheckSQLiteReadyPasses -count=1

test:
	go test ./...

test-race:
	go test -race ./...

vet:
	go vet ./...

analyze:
	staticcheck ./...
	govulncheck ./...
