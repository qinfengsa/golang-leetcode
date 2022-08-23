package dsu

import "sort"

type Dsu struct {
	parent []int
}

func createDsu(n int) *Dsu {
	parent := make([]int, n)
	for i := 0; i < n; i++ {
		parent[i] = i
	}
	return &Dsu{parent: parent}
}

func (this *Dsu) findParent(num int) int {
	parent := this.parent[num]
	if num == parent {
		return num
	}
	return this.findParent(parent)
}

func (this *Dsu) union(num1, num2 int) {
	num1, num2 = this.findParent(num1), this.findParent(num2)
	this.parent[num1] = num2
}

// 721. 账户合并
// 给定一个列表 accounts，每个元素 accounts[i] 是一个字符串列表，其中第一个元素 accounts[i][0] 是 名称 (name)，其余元素是 emails 表示该账户的邮箱地址。
//
// 现在，我们想合并这些账户。如果两个账户都有一些共同的邮箱地址，则两个账户必定属于同一个人。请注意，即使两个账户具有相同的名称，它们也可能属于不同的人，因为人们可能具有相同的名称。一个人最初可以拥有任意数量的账户，但其所有账户都具有相同的名称。
//
// 合并账户后，按以下格式返回账户：每个账户的第一个元素是名称，其余元素是 按字符 ASCII 顺序排列 的邮箱地址。账户本身可以以 任意顺序 返回。
//
// 示例 1：
// 输入：accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
// 输出：[["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
// 解释：
// 第一个和第三个 John 是同一个人，因为他们有共同的邮箱地址 "johnsmith@mail.com"。
// 第二个 John 和 Mary 是不同的人，因为他们的邮箱地址没有被其他帐户使用。
// 可以以任何顺序返回这些列表，例如答案 [['Mary'，'mary@mail.com']，['John'，'johnnybravo@mail.com']，
// ['John'，'john00@mail.com'，'john_newyork@mail.com'，'johnsmith@mail.com']] 也是正确的。
//
// 示例 2：
// 输入：accounts = [["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"],["Ethan","Ethan5@m.co","Ethan4@m.co","Ethan0@m.co"],["Hanzo","Hanzo3@m.co","Hanzo1@m.co","Hanzo0@m.co"],["Fern","Fern5@m.co","Fern1@m.co","Fern0@m.co"]]
// 输出：[["Ethan","Ethan0@m.co","Ethan4@m.co","Ethan5@m.co"],["Gabe","Gabe0@m.co","Gabe1@m.co","Gabe3@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co","Hanzo3@m.co"],["Kevin","Kevin0@m.co","Kevin3@m.co","Kevin5@m.co"],["Fern","Fern0@m.co","Fern1@m.co","Fern5@m.co"]]
//
// 提示：
// 1 <= accounts.length <= 1000
// 2 <= accounts[i].length <= 10
// 1 <= accounts[i][j].length <= 30
// accounts[i][0] 由英文字母组成
// accounts[i][j] (for j > 0) 是有效的邮箱地址
func accountsMerge(accounts [][]string) [][]string {
	dsu := createDsu(10001)
	id := 0
	result := make([][]string, 0)
	email2Id := make(map[string]int)
	email2Name := make(map[string]string)

	for _, account := range accounts {
		name, firstEmail := account[0], account[1]
		email2Name[firstEmail] = name
		if _, ok := email2Id[firstEmail]; !ok {
			email2Id[firstEmail] = id
			id++
		}

		for i := 2; i < len(account); i++ {
			email := account[i]
			email2Name[email] = name
			if _, ok := email2Id[email]; !ok {
				email2Id[email] = id
				id++
			}
			dsu.union(email2Id[firstEmail], email2Id[email])
		}
	}
	id2EmailList := make(map[int][]string)
	for email := range email2Name {
		parentId := dsu.findParent(email2Id[email])
		if _, ok := id2EmailList[parentId]; !ok {
			id2EmailList[parentId] = make([]string, 0)
		}
		id2EmailList[parentId] = append(id2EmailList[parentId], email)
	}
	for _, emailList := range id2EmailList {
		sort.Strings(emailList)
		name := email2Name[emailList[0]]
		result = append(result, append([]string{name}, emailList...))
	}

	return result
}

// 765. 情侣牵手
// n 对情侣坐在连续排列的 2n 个座位上，想要牵到对方的手。
//
// 人和座位由一个整数数组 row 表示，其中 row[i] 是坐在第 i 个座位上的人的 ID。情侣们按顺序编号，第一对是 (0, 1)，第二对是 (2, 3)，以此类推，最后一对是 (2n-2, 2n-1)。
//
// 返回 最少交换座位的次数，以便每对情侣可以并肩坐在一起。 每次交换可选择任意两人，让他们站起来交换座位。
//
// 示例 1:
// 输入: row = [0,2,1,3]
// 输出: 1
// 解释: 只需要交换row[1]和row[2]的位置即可。
//
// 示例 2:
// 输入: row = [3,2,0,1]
// 输出: 0
// 解释: 无需交换座位，所有的情侣都已经可以手牵手了。
//
// 提示:
// 2n == row.length
// 2 <= n <= 30
// n 是偶数
// 0 <= row[i] < 2n
// row 中所有元素均无重复
func minSwapsCouples(row []int) int {
	n := len(row) >> 1
	dsu := createDsu(n)
	for i := 0; i < len(row); i += 2 {
		dsu.union(row[i]>>1, row[i+1]>>1)
	}
	circle := 0
	for i := 0; i < n; i++ {
		if i == dsu.findParent(i) {
			// 存在环
			circle++
		}
	}
	return n - circle
}
