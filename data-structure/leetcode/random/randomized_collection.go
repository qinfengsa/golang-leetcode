package random

import "math/rand"

// RandomizedCollection
// 381. O(1) 时间插入、删除和获取随机元素 - 允许重复
// 设计一个支持在平均 时间复杂度 O(1) 下， 执行以下操作的数据结构。
//
// 注意: 允许出现重复元素。
// insert(val)：向集合中插入元素 val。
// remove(val)：当 val 存在时，从集合中移除一个 val。
// getRandom：从现有集合中随机获取一个元素。每个元素被返回的概率应该与其在集合中的数量呈线性相关。
//
// 示例:
// // 初始化一个空的集合。
// RandomizedCollection collection = new RandomizedCollection();
// // 向集合中插入 1 。返回 true 表示集合不包含 1 。
// collection.insert(1);
// // 向集合中插入另一个 1 。返回 false 表示集合包含 1 。集合现在包含 [1,1] 。
// collection.insert(1);
// // 向集合中插入 2 ，返回 true 。集合现在包含 [1,1,2] 。
// collection.insert(2);
// // getRandom 应当有 2/3 的概率返回 1 ，1/3 的概率返回 2 。
// collection.getRandom();
// // 从集合中删除 1 ，返回 true 。集合现在包含 [1,2] 。
// collection.remove(1);
// // getRandom 应有相同概率返回 1 和 2 。
// collection.getRandom();
type RandomizedCollection struct {
	indexMap map[int]map[int]bool
	nums     []int
}

func Constructor2() RandomizedCollection {
	return RandomizedCollection{indexMap: map[int]map[int]bool{}, nums: []int{}}
}

// Insert 向集合中插入元素 val
func (this *RandomizedCollection) Insert(val int) bool {

	size := len(this.nums)
	indexs := make(map[int]bool)
	if v, ok := this.indexMap[val]; ok {
		indexs = v
	} else {
		this.indexMap[val] = indexs
	}
	result := len(indexs) == 0
	indexs[size] = true
	this.nums = append(this.nums, val)
	return result

}

// Remove 当 val 存在时，从集合中移除一个 val。
func (this *RandomizedCollection) Remove(val int) bool {
	if idxs, ok := this.indexMap[val]; ok {
		idxSize := len(idxs)
		if idxSize == 0 {
			return false
		}
		size := len(this.nums)
		swapVal := this.nums[size-1]
		var idx int
		// 当前 val 在 num 中 的 位置idx
		for i := range idxs {
			idx = i
			break
		}
		this.nums = this.nums[:size-1]
		delete(idxs, idx)
		if idx != size-1 {
			this.nums[idx] = swapVal
			// 原来的size-1 所在的位置 需要移除
			swapIndexs := this.indexMap[swapVal]
			// 原来的size-1需要移除
			delete(swapIndexs, size-1)
			swapIndexs[idx] = true
		}

		return true
	}
	return false
}

// GetRandom 从现有集合中随机获取一个元素。每个元素被返回的概率应该与其在集合中的数量呈线性相关。
func (this *RandomizedCollection) GetRandom() int {
	size := len(this.nums)
	idx := rand.Intn(size)
	return this.nums[idx]
}
