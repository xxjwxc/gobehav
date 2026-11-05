package matcher

import (
	"reflect"
	"sort"
	"testing"
)

func find(patterns []string, text string) []Match {
	out := New(patterns).FindAll([]rune(text))
	sort.Slice(out, func(i, j int) bool {
		if out[i].Start != out[j].Start {
			return out[i].Start < out[j].Start
		}
		return out[i].End < out[j].End
	})
	return out
}

func TestFindAll(t *testing.T) {
	got := find([]string{"对不起", "对不起我", "抱歉"}, "对不起我很抱歉")
	want := []Match{
		{Start: 0, End: 2, Index: 0},
		{Start: 0, End: 3, Index: 1},
		{Start: 5, End: 6, Index: 2},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

// TestOverlap 交错命中：ushers = u s h e r s，同时命中 she / he / hers。
func TestOverlap(t *testing.T) {
	got := find([]string{"he", "she", "hers"}, "ushers")
	want := []Match{
		{Start: 1, End: 3, Index: 1}, // she
		{Start: 2, End: 3, Index: 0}, // he
		{Start: 2, End: 5, Index: 2}, // hers
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestEmpty(t *testing.T) {
	if got := New(nil).FindAll([]rune("abc")); len(got) != 0 {
		t.Fatalf("no pattern should match nothing, got %+v", got)
	}
	if got := New([]string{""}).FindAll([]rune("abc")); len(got) != 0 {
		t.Fatalf("empty pattern should be ignored, got %+v", got)
	}
	if got := New([]string{"a"}).FindAll(nil); len(got) != 0 {
		t.Fatalf("empty text should match nothing, got %+v", got)
	}
}
