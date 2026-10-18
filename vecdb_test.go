package vecdb

import (
	"fmt"
	"math/rand"
	"testing"
)

func randVec(r *rand.Rand, d int) []float64 {
	v := make([]float64, d)
	for i := range v {
		v[i] = r.NormFloat64()
	}
	return v
}

func TestExactSearchFindsSelf(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	ix := New(16, 0, 0, 0)
	vs := make([][]float64, 200)
	for i := range vs {
		vs[i] = randVec(r, 16)
		ix.Add(fmt.Sprint(i), vs[i])
	}
	for i := 0; i < 200; i += 17 {
		res, _ := ix.Search(vs[i], 3)
		if res[0].ID != fmt.Sprint(i) || res[0].Score < 0.9999 {
			t.Fatalf("query %d got %+v", i, res[0])
		}
		if res[0].Score < res[1].Score {
			t.Fatal("results not sorted")
		}
	}
}

func TestErrors(t *testing.T) {
	ix := New(3, 0, 0, 0)
	if ix.Add("a", []float64{1, 2}) == nil || ix.Add("b", []float64{0, 0, 0}) == nil {
		t.Fatal("expected errors")
	}
}

func TestApproxRecall(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	ix := New(32, 12, 10, 3)
	vs := make([][]float64, 2000)
	for i := range vs {
		vs[i] = randVec(r, 32)
		ix.Add(fmt.Sprint(i), vs[i])
	}
	hits := 0
	for q := 0; q < 100; q++ {
		// a slightly perturbed stored vector: its true nearest neighbour is the original
		base := vs[q*7]
		qv := make([]float64, 32)
		for i := range qv {
			qv[i] = base[i] + 0.1*r.NormFloat64()
		}
		exact, _ := ix.Search(qv, 1)
		approx, _ := ix.ApproxSearch(qv, 1)
		if len(approx) > 0 && approx[0].ID == exact[0].ID {
			hits++
		}
	}
	if hits < 90 {
		t.Fatalf("recall@1 too low: %d/100", hits)
	}
}

func BenchmarkExact(b *testing.B)  { benchSearch(b, false) }
func BenchmarkApprox(b *testing.B) { benchSearch(b, true) }

func benchSearch(b *testing.B, approx bool) {
	r := rand.New(rand.NewSource(4))
	ix := New(64, 8, 12, 5)
	for i := 0; i < 20000; i++ {
		ix.Add(fmt.Sprint(i), randVec(r, 64))
	}
	q := randVec(r, 64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if approx {
			ix.ApproxSearch(q, 10)
		} else {
			ix.Search(q, 10)
		}
	}
}
