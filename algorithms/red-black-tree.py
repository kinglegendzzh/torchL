class Node:
    def __init__(self, data, color="red"):
        self.data = data
        self.color = color  # 节点颜色
        self.left = None
        self.right = None
        self.parent = None


class RedBlackTree:
    def __init__(self):
        self.TNULL = Node(0, color="black")
        self.root = self.TNULL

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def insert(self, key):
        node = Node(key)
        node.left = self.TNULL
        node.right = self.TNULL

        y = None
        x = self.root

        while x != self.TNULL:
            y = x
            if node.data < x.data:
                x = x.left
            else:
                x = x.right

        node.parent = y
        if y is None:
            self.root = node
        elif node.data < y.data:
            y.left = node
        else:
            y.right = node

        node.color = "red"
        self.fix_insert(node)

    def fix_insert(self, k):
        while k.parent is not None and k.parent.color == "red":
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u.color == "red":
                    u.color = "black"
                    k.parent.color = "black"
                    k.parent.parent.color = "red"
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = "black"
                    k.parent.parent.color = "red"
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right
                if u.color == "red":
                    u.color = "black"
                    k.parent.color = "black"
                    k.parent.parent.color = "red"
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = "black"
                    k.parent.parent.color = "red"
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = "black"

    def __repr__(self):
        return self._print_helper(self.root, "", True)

    def _print_helper(self, node, indent, last):
        if node != self.TNULL:
            s_color = "RED" if node.color == "red" else "BLACK"
            result = f"{indent} {'R----' if last else 'L----'} {node.data}({s_color})\n"
            indent += "     " if last else "|    "
            result += self._print_helper(node.left, indent, False)
            result += self._print_helper(node.right, indent, True)
            return result
        return ""


# 示例代码
if __name__ == "__main__":
    rbt = RedBlackTree()
    rbt.insert(55)
    rbt.insert(40)
    rbt.insert(65)
    rbt.insert(60)
    rbt.insert(75)
    rbt.insert(57)
    rbt.insert(66)
    rbt.insert(67)
    rbt.insert(68)
    rbt.insert(69)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    rbt.insert(70)
    print(rbt)
