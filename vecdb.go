// Package vecdb is a small in-memory vector database with exact (flat)
// and approximate (random-hyperplane LSH) cosine-similarity search.
package vecdb

import (
	"container/heap"
	"errors"
	"math"
	"math/rand"
	"sort"
)

// Result is a search hit.
type Result struct {
	ID    string
	Score float64 // cosine similarity
}

type entry struct {
	id  string
	vec []float64 // unit length
}

// Index stores normalised vectors and answers nearest-neighbour queries.
type Index struct {
	dim     int
	entries []entry
	planes  [][][]float64 // [table][bit][dim]
	buckets []map[uint32][]int
}

// New creates an index. tables and bits configure the LSH; set tables to 0 for flat search only.
func New(dim, tables, bits int, seed int64) *Index {
	ix := &Index{dim: dim}
	rng := rand.New(rand.NewSource(seed))
	for t := 0; t < tables; t++ {
		p := make([][]float64, bits)
		for b := range p {
			p[b] = make([]float64, dim)
			for d := range p[b] {
				p[b][d] = rng.NormFloat64()
			}
		}
		ix.planes = append(ix.planes, p)
		ix.buckets = append(ix.buckets, map[uint32][]int{})
	}
	return ix
}

func normalise(v []float64) ([]float64, error) {
	var s float64
	for _, x := range v {
		s += x * x
	}
	if s == 0 {
		return nil, errors.New("vecdb: zero vector")
	}
	n := math.Sqrt(s)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / n
	}
	return out, nil
}

func dot(a, b []float64) (s float64) {
	for i := range a {
		s += a[i] * b[i]
	}
	return
}

func (ix *Index) hash(t int, v []float64) (h uint32) {
	for b, p := range ix.planes[t] {
		if dot(p, v) >= 0 {
			h |= 1 << uint(b)
		}
	}
	return
}

// Add inserts a vector under id.
func (ix *Index) Add(id string, v []float64) error {
	if len(v) != ix.dim {
		return errors.New("vecdb: dimension mismatch")
	}
	u, err := normalise(v)
	if err != nil {
		return err
	}
	pos := len(ix.entries)
	ix.entries = append(ix.entries, entry{id, u})
	for t := range ix.planes {
		h := ix.hash(t, u)
		ix.buckets[t][h] = append(ix.buckets[t][h], pos)
	}
	return nil
}

// Len returns the number of stored vectors.
func (ix *Index) Len() int { return len(ix.entries) }

type minHeap []Result

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].Score < h[j].Score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(Result)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

func (ix *Index) topK(q []float64, cand []int, k int) []Result {
	h := &minHeap{}
	for _, i := range cand {
		r := Result{ix.entries[i].id, dot(q, ix.entries[i].vec)}
		if h.Len() < k {
			heap.Push(h, r)
		} else if r.Score > (*h)[0].Score {
			(*h)[0] = r
			heap.Fix(h, 0)
		}
	}
	out := append([]Result(nil), *h...)
	sort.Slice(out, func(a, b int) bool { return out[a].Score > out[b].Score })
	return out
}

// Search returns the exact top-k most similar vectors (O(n·d)).
func (ix *Index) Search(v []float64, k int) ([]Result, error) {
	q, err := normalise(v)
	if err != nil {
		return nil, err
	}
	all := make([]int, len(ix.entries))
	for i := range all {
		all[i] = i
	}
	return ix.topK(q, all, k), nil
}

// ApproxSearch ranks only the vectors that share an LSH bucket with the query in at least one table.
func (ix *Index) ApproxSearch(v []float64, k int) ([]Result, error) {
	q, err := normalise(v)
	if err != nil {
		return nil, err
	}
	seen := map[int]bool{}
	var cand []int
	for t := range ix.planes {
		for _, i := range ix.buckets[t][ix.hash(t, q)] {
			if !seen[i] {
				seen[i] = true
				cand = append(cand, i)
			}
		}
	}
	return ix.topK(q, cand, k), nil
}
