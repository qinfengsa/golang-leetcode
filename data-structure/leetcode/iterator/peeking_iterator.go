package iterator

type Iterator struct {
}

func (this *Iterator) hasNext() bool {
	// Returns true if the iteration has more elements.
	return false
}

func (this *Iterator) next() int {
	// Returns the next element in the iteration.
	return 0
}

type PeekingIterator struct {
	nums  []int
	index int
}

func Constructor(iter *Iterator) *PeekingIterator {
	nums := make([]int, 0)
	for iter.hasNext() {
		nums = append(nums, iter.next())
	}
	return &PeekingIterator{
		nums:  nums,
		index: 0,
	}
}

func (this *PeekingIterator) hasNext() bool {
	n := len(this.nums)
	return this.index < n
}

func (this *PeekingIterator) next() int {
	n := len(this.nums)
	if this.index >= n {
		return -1
	}
	val := this.nums[this.index]
	this.index++
	return val
}

func (this *PeekingIterator) peek() int {
	n := len(this.nums)
	if this.index >= n {
		return -1
	}
	val := this.nums[this.index]
	return val
}
