package store

import (
	"errors"
	"fmt"
)

// MissingIndexError reports that the requested index file does not exist yet.
type MissingIndexError struct {
	Path string
}

func (e *MissingIndexError) Error() string {
	return fmt.Sprintf("index %s does not exist; rebuild the stroma index first", e.Path)
}

// IsMissingIndex reports whether err wraps a missing-index failure.
func IsMissingIndex(err error) bool {
	var target *MissingIndexError
	return errors.As(err, &target)
}

// MissingIndexPath returns the configured path for a missing-index failure.
func MissingIndexPath(err error) string {
	var target *MissingIndexError
	if errors.As(err, &target) {
		return target.Path
	}
	return ""
}
