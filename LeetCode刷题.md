# LeetCode刷题

[toc]

## 学习资源

刷题教程: [leetcode-master](https://github.com/youngyangyang04/leetcode-master)

## Background Knowledge

### 复杂度

![](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/uPic/image-20221027061820654.png)![image-20221027061842225](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/uPic/image-20221027061842225.png)

![image-20221027061852249](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/uPic/image-20221027061852249.png)

## Data Structure

### Array

### Binary Search

应用条件:

1. 有序数组
2. 无重复元素

两种写法:

1. 左闭右闭 [left, right]

   ```java
   public static Integer binary_search_v1(ArrayList<Integer> nums, Integer target){
           int left=0;
           int right=nums.size()-1;
           while (left<=right){
               int middle=left+((right-left)/2);
               if (nums.get(middle) >target){
                   right=middle-1;
               }else if (nums.get(middle) <target){
                   left=middle+1;
               }else{
                   return middle;
               }
           }
           return -1;
       }
   ```
2. 左闭右开 [left, right)

   ```java
   public static Integer binary_search_v2(ArrayList<Integer> nums, Integer target){
           int left=0;
           int right=nums.size()-1;
           while (left<right){
               int middle=left+((right-left)/2);
               if (nums.get(middle) >target){
                   right=middle-1;
               }else if (nums.get(middle) <target){
                   left=middle+1;
               }else{
                   return middle;
               }
           }
           return -1;
       }
   ```

### Two Pointers

```java
public static int two_pointers(int[] nums, int val) {

        int fastIndex = 0;
        int slowIndex;
        for (slowIndex = 0; fastIndex < nums.length; fastIndex++) {
            if (nums[fastIndex] != val) {
                nums[slowIndex] = nums[fastIndex];
                slowIndex++;
            }
        }
        return slowIndex;

    }
```

#### Three Pointers

this scructure include two pointers in nums and another pointer for new list position

Leetcode 977

```java
public int[] three_pointers(int[] nums) {
        int right = nums.length - 1;
        int left = 0;
        int[] result = new int[nums.length];
        int index = result.length - 1;
        while (left <= right) {
            if (nums[left] * nums[left] > nums[right] * nums[right]) {
                result[index--] = nums[left] * nums[left];
                ++left;
            } else {
                result[index--] = nums[right] * nums[right];
                --right;
            }
        }
        return result;
    }
```

**O(n)**

### Sliding Window

Leetcode 209

```java
public int sliding_window(int s, int[] nums) {
        int left = 0;
        int sum = 0;
        int result = Integer.MAX_VALUE;
        for (int right = 0; right < nums.length; right++) {
            sum += nums[right];
            while (sum >= s) {
                result = Math.min(result, right - left + 1);
                sum -= nums[left++];
            }
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }
```

Although "while" inside the "for" loop, each ele actually been processed only twice. Therefore, O(2n)=O(n)

**O(n)**

**弱智顺时针转圈, 我就是不做, 咬死我**

Leetcode 59

### Link List

Leetcode 237 141 92 25

| -         | Insert | Lookup | Suit for                                                            |
| --------- | ------ | ------ | ------------------------------------------------------------------- |
| Array     | O(n)   | O(1)   | based on its complexisity, suit for situation whih need more lookup |
| Link List | O(1)   | O(n)   | In the same way, insert                                             |

#### Remove ele

with new header

```java
/**
 * 添加虚节点方式
 * 时间复杂度 O(n)
 * 空间复杂度 O(1)
 * @param head
 * @param val
 * @return
 */
public ListNode removeElements(ListNode head, int val) {
    if (head == null) {
        return head;
    }
    // 因为删除可能涉及到头节点，所以设置dummy节点，统一操作
    ListNode dummy = new ListNode(-1, head);
    ListNode pre = dummy;
    ListNode cur = head;
    while (cur != null) {
        if (cur.val == val) {
            pre.next = cur.next;
        } else {
            pre = cur;
        }
        cur = cur.next;
    }
    return dummy.next;
}
```

without new header

```java
/**
 * 不添加虚拟节点方式
 * 时间复杂度 O(n)
 * 空间复杂度 O(1)
 * @param head
 * @param val
 * @return
 */
public ListNode removeElements(ListNode head, int val) {
    while (head != null && head.val == val) {
        head = head.next;
    }
    // 已经为null，提前退出
    if (head == null) {
        return head;
    }
    // 已确定当前head.val != val
    ListNode pre = head;
    ListNode cur = head.next;
    while (cur != null) {
        if (cur.val == val) {
            pre.next = cur.next;
        } else {
            pre = cur;
        }
        cur = cur.next;
    }
    return head;
}
```

#### Full linked list

```java
//单链表
class ListNode {
int val;
ListNode next;
ListNode(){}
ListNode(int val) {
this.val=val;
}
}
class MyLinkedList {
    //size存储链表元素的个数
    int size;
    //虚拟头结点
    ListNode head;

    //初始化链表
    public MyLinkedList() {
        size = 0;
        head = new ListNode(0);
    }

    //获取第index个节点的数值
    public int get(int index) {
        //如果index非法，返回-1
        if (index < 0 || index >= size) {
            return -1;
        }
        ListNode currentNode = head;
        //包含一个虚拟头节点，所以查找第 index+1 个节点
        for (int i = 0; i <= index; i++) {
            currentNode = currentNode.next;
        }
        return currentNode.val;
    }

    //在链表最前面插入一个节点
    public void addAtHead(int val) {
        addAtIndex(0, val);
    }

    //在链表的最后插入一个节点
    public void addAtTail(int val) {
        addAtIndex(size, val);
    }

    // 在第 index 个节点之前插入一个新节点，例如index为0，那么新插入的节点为链表的新头节点。
    // 如果 index 等于链表的长度，则说明是新插入的节点为链表的尾结点
    // 如果 index 大于链表的长度，则返回空
    public void addAtIndex(int index, int val) {
        if (index > size) {
            return;
        }
        if (index < 0) {
            index = 0;
        }
        size++;
        //找到要插入节点的前驱
        ListNode pred = head;
        for (int i = 0; i < index; i++) {
            pred = pred.next;
        }
        ListNode toAdd = new ListNode(val);
        toAdd.next = pred.next;
        pred.next = toAdd;
    }

    //删除第index个节点
    public void deleteAtIndex(int index) {
        if (index < 0 || index >= size) {
            return;
        }
        size--;
        ListNode pred = head;
        for (int i = 0; i < index; i++) {
            pred = pred.next;
        }
        pred.next = pred.next.next;
    }
}

//双链表
class MyLinkedList {
    class ListNode {
        int val;
        ListNode next,prev;
        ListNode(int x) {val = x;}
    }

    int size;
    ListNode head,tail;//Sentinel node

    /** Initialize your data structure here. */
    public MyLinkedList() {
        size = 0;
        head = new ListNode(0);
        tail = new ListNode(0);
        head.next = tail;
        tail.prev = head;
    }

    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    public int get(int index) {
        if(index < 0 || index >= size){return -1;}
        ListNode cur = head;

        // 通过判断 index < (size - 1) / 2 来决定是从头结点还是尾节点遍历，提高效率
        if(index < (size - 1) / 2){
            for(int i = 0; i <= index; i++){
                cur = cur.next;
            }          
        }else{
            cur = tail;
            for(int i = 0; i <= size - index - 1; i++){
                cur = cur.prev;
            }
        }
        return cur.val;
    }

    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    public void addAtHead(int val) {
        ListNode cur = head;
        ListNode newNode = new ListNode(val);
        newNode.next = cur.next;
        cur.next.prev = newNode;
        cur.next = newNode;
        newNode.prev = cur;
        size++;
    }

    /** Append a node of value val to the last element of the linked list. */
    public void addAtTail(int val) {
        ListNode cur = tail;
        ListNode newNode = new ListNode(val);
        newNode.next = tail;
        newNode.prev = cur.prev;
        cur.prev.next = newNode;
        cur.prev = newNode;
        size++;
    }

    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    public void addAtIndex(int index, int val) {
        if(index > size){return;}
        if(index < 0){index = 0;}
        ListNode cur = head;
        for(int i = 0; i < index; i++){
            cur = cur.next;
        }
        ListNode newNode = new ListNode(val);
        newNode.next = cur.next;
        cur.next.prev = newNode;
        newNode.prev = cur;
        cur.next = newNode;
        size++;
    }

    /** Delete the index-th node in the linked list, if the index is valid. */
    public void deleteAtIndex(int index) {
        if(index >= size || index < 0){return;}
        ListNode cur = head;
        for(int i = 0; i < index; i++){
            cur = cur.next;
        }
        cur.next.next.prev = cur;
        cur.next = cur.next.next;
        size--;
    }
}

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList obj = new MyLinkedList();
 * int param_1 = obj.get(index);
 * obj.addAtHead(val);
 * obj.addAtTail(val);
 * obj.addAtIndex(index,val);
 * obj.deleteAtIndex(index);
 */
```

#### Reverse single linked list

* loop

  ```java
  // 双指针
  class Solution {
      public ListNode reverseList(ListNode head) {
          ListNode prev = null;
          ListNode cur = head;
          ListNode temp = null;
          while (cur != null) {
              temp = cur.next;// 保存下一个节点
              cur.next = prev;
              prev = cur;
              cur = temp;
          }
          return prev;
      }
  }
  ```
* recursive

  ```java
  // 递归 
  class Solution {
      public ListNode reverseList(ListNode head) {
          return reverse(null, head);
      }
  
      private ListNode reverse(ListNode prev, ListNode cur) {
          if (cur == null) {
              return prev;
          }
          ListNode temp = null;
          temp = cur.next;// 先保存下一个节点
          cur.next = prev;// 反转
          // 更新prev、cur位置
          prev = cur;
          cur = temp;
          return reverse(prev, cur);
      }
  ```

### Heap

**minHeap&&maxHeap**

```java
PriorityQueue<int> minHeap = new  PriorityQueue<>((a,b)->a-b);
PriorityQueue<int> maxHeap = new  PriorityQueue<>((a,b)->b-a);
```



**Background**

Push, pop

具体来说：

- 如果永远都维护一个有序数组的方式取极值很容易，但是插队麻烦。
- 如果永远都维护一个有序链表的方式取极值也容易。 不过要想查找足够快，而不是线性扫描，就需要借助索引，这种实现对应的就是优先级队列的**跳表实现**。
- 如果永远都维护一个树的方式取极值也可以实现，比如根节点就是极值，这样 O(1) 也可以取到极值，但是调整过程需要 $O(logN)$。这种实现对应的就是优先级队列的**二叉堆实现**。

简单总结下就是，**堆就是动态帮你求极值的**。当你需要动态求最大或最小值就就用它。而具体怎么实现，复杂度的分析我们之后讲，现在你只要记住使用场景，堆是如何解决这些问题的以及堆的 api 就够了。



**API**

poll() peek and remove top value

peek() view the top value

add() inset the value to heap and sort, only O(logn)

offer() same as add()

minHeap-> minmum value at the top

maxHeap-> maximum value at the top

maxHeap-> find kth Largest element->there are k elements bigger than the kth->



template 

[215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/description/)

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minHeap=new PriorityQueue<>();
        for(int i=0;i<k;i++){
            minHeap.add(nums[i]);
        }
        for(int i=k;i<nums.length;i++){
            if (nums[i]>minHeap.peek()){
                minHeap.poll();
                minHeap.add(nums[i]);
            }
        }
        return minHeap.peek();

    }
}
class Solution {
# It's an online algorithm since each time the top of heap is the kth largest element, and
#  dit's scalable if the input is a stream
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minHeap=new PriorityQueue<>();
        for(int x:nums){
          if (minHeap.size()<k||x+=minHeap.peek()){
            minHeap.offer(x);
          }
          if (minHeap.size()>k){
            minHeap.poll();
          }
        return heap.peek()
    }
}
```

[23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/description/)

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode>minHeap=new PriorityQueue<>((a,b)->a.val-b.val);
        for(ListNode Nodehead:lists){
            if (Nodehead!=null){
                minHeap.offer(Nodehead);
            }
        }
        ListNode head=new ListNode(),cur=head;
        while(!minHeap.isEmpty()){
            ListNode top=minHeap.poll();
            cur.next=top;
            cur=top;
            if (top.next!=null){
                minHeap.offer(top.next);
            }
        }
        return head.next;
    }
}
```

- Sorting O(nlogn)
  - Sort the array and get the kth element
- Max Heap O(n+klogn)
  - Heapify the array, then poll $\mathrm{k}$ times
- Min Heap O(nlogk)
  - Maintain a size $\mathrm{k}$ min heap
  - Add element to min heap if greater than or equal to the top element, adjust size if necessary
  - At the end top of heap is the kth largest



### HashMap

当调用hashmap.keys()时, 返回的key实际是按照hashcode排序的

## Leetcode

### 509 Fibonacci Number

递归, 基于公式

> ```
> F(0) = 0, F(1) = 1
> F(n) = F(n - 1) + F(n - 2), for n > 1
> ```



