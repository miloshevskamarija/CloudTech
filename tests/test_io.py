import pytest
pytest.importorskip("uproot")
from src.analysis import io

class DummyTree:
    classname = "TTree"

class DummyFileMini:
    # Mimic a ROOT file that has a 'mini' TTree

    def __init__(self):
        self._store = {"mini": DummyTree()}
        self.file_path = "dummy.root"

    def keys(self):
        return list(self._store.keys())

    def __getitem__(self, key):
        return self._store[key]

    def classnames(self):
        return {}

class DummyFileUnique:
    # Mimic a ROOT file with exactly one TTree at the top level

    def __init__(self):
        self._store = {"mytree": DummyTree()}
        self.file_path = "unique.root"

    def keys(self):
        return list(self._store.keys())

    def __getitem__(self, key):
        return self._store[key]

    def classnames(self):
        return {name: obj.classname for name, obj in self._store.items()}

class DummyDir:
    def __init__(self, children):
        self._children = children

    def keys(self):
        return list(self._children.keys())

    def __getitem__(self, key):
        return self._children[key]

class DummyFileNested:
    # Mimic a ROOT file where the TTree lives inside a directory

    def __init__(self):
        tree = DummyTree()
        self.file_path = "nested.root"
        self._store = {
            "dir1": DummyDir({"subtree": tree}),
            "dir1/subtree": tree,
        }

    def keys(self):
        # Only top-level keys, like uproot
        return [k for k in self._store.keys() if "/" not in k]

    def __getitem__(self, key):
        return self._store[key]

    def classnames(self):
        # No top-level TTrees in this scenario
        return {}

def test_find_tree_prefers_mini_key():
    f = DummyFileMini()
    tree = io._find_tree(f)
    assert isinstance(tree, DummyTree)

def test_find_tree_unique_ttree_via_classnames():
    f = DummyFileUnique()
    tree = io._find_tree(f)
    assert isinstance(tree, DummyTree)

def test_find_tree_nested_directory_search():
    f = DummyFileNested()
    tree = io._find_tree(f)
    assert isinstance(tree, DummyTree)
