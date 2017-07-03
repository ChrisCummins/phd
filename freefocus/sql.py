"""
SQL schema for FreeFocus.
"""
import sqlalchemy as sql

from datetime import datetime
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import String
from sqlalchemy import UnicodeText
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship
from typing import List, Dict, Tuple


Base = declarative_base()


def get_or_create(session: sql.orm.session.Session, model,
                  defaults: Dict[str, object]=None, **kwargs) -> object:
    """
    Instantiate a mapped database object. If the object is not in the database,
    add it.
    """
    instance = session.query(model).filter_by(**kwargs).first()

    if not instance:
        params = dict((k, v) for k, v in kwargs.items()
                      if not isinstance(v, sql.sql.expression.ClauseElement))
        params.update(defaults or {})
        instance = model(**params)
        session.add(instance)

    return instance



class Meta(Base):
    __tablename__ = "meta"

    key = Column(String(255), primary_key=True)
    value = Column(String(255), nullable=False)


### Persons


class Person(Base):
    __tablename__ = "persons"

    uid = Column(Integer, primary_key=True)
    name = Column(UnicodeText(length=255), nullable=False)

    emails = relationship("Email")
    groups = relationship("Group", secondary="person_group_associations")

    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    def json(self):
        return {
            "uid": self.uid,
            "name": self.name,
            "created": str(self.created),
        }


class Email(Base):
    __tablename__ = "email_addresses"

    person_uid = Column(Integer, ForeignKey("persons.uid"),
                        nullable=False)
    person = relationship("Person")
    address = Column(String(255), nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('person_uid', 'address', name='_uid'),)


### Workspace


class Workspace(Base):
    """ only one Workspace per database """
    __tablename__ = "workspaces"

    uid = Column(String(255), primary_key=True)
    created = Column(DateTime, nullable=False, default=datetime.utcnow)

    owners = relationship(
        "Group", secondary="workspace_owner_associations",
        primaryjoin="WorkspaceOwnerAssociation.workspace_uid == Group.id",
        secondaryjoin="WorkspaceOwnerAssociation.owner_id == Group.id")

    friends = relationship(
        "Group", secondary="workspace_friend_associations",
        primaryjoin="WorkspaceFriendAssociation.workspace_uid == Group.id",
        secondaryjoin="WorkspaceFriendAssociation.friend_id == Group.id")

    comments = relationship("WorkspaceComment")


class WorkspaceOwnerAssociation(Base):
    __tablename__ = "workspace_owner_associations"
    workspace_uid = Column(Integer, ForeignKey("workspaces.uid"), nullable=False)
    owner_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('workspace_uid', 'owner_id', name='_uid'),)


class WorkspaceFriendAssociation(Base):
    __tablename__ = "workspace_friend_associations"
    workspace_uid = Column(Integer, ForeignKey("workspaces.uid"), nullable=False)
    friend_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('workspace_uid', 'friend_id', name='_uid'),)


class WorkspaceComment(Base):
    __tablename__ = "workspace_comments"
    id = Column(Integer, primary_key=True)

    # a workspace comment's parent is either a workspace or another comment
    workspace_uid = Column(Integer, ForeignKey("workspaces.uid"))
    workspace = relationship("Workspace")
    parent_id = Column(Integer, ForeignKey("workspace_comments.id"))
    children = relationship(
        "WorkspaceComment", backref=backref('parent', remote_side=[id]))

    body = Column(UnicodeText(length=2**31), nullable=False)

    # Accountability
    created_by_id = Column(Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship("Person")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified = Column(DateTime)  # comments may only be modified by the creator


### Groups


class Group(Base):
    __tablename__ = "groups"

    id = Column(Integer, primary_key=True)

    # null parent ID means the group belongs to the workspace.
    parent_id = Column(Integer, ForeignKey("groups.id"))
    children = relationship(
        "Group", backref=backref('parent', remote_side=[id]))

    owners = relationship(
        "Group", secondary="group_owner_associations",
        primaryjoin="GroupOwnerAssociation.group_id == Group.id",
        secondaryjoin="GroupOwnerAssociation.owner_id == Group.id")

    friends = relationship(
        "Group", secondary="group_friend_associations",
        primaryjoin="GroupFriendAssociation.group_id == Group.id",
        secondaryjoin="GroupFriendAssociation.friend_id == Group.id")

    body = Column(UnicodeText(length=2**31), nullable=False)

    members = relationship("Person", secondary="person_group_associations")

    # Accountability
    created_by_id = Column(
        Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship(
        "Person", primaryjoin="Person.uid == Group.created_by_id")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified_by_id = Column(Integer, ForeignKey("persons.uid"))
    modified_by = relationship(
        "Person", primaryjoin="Person.uid == Group.modified_by_id")
    modified = Column(DateTime)

    comments = relationship("GroupComment")

    def validate(self):
        for owner in self.owners:
            if self.id == owner.id:
                raise ValueError

        for friend in self.friends:
            if self.id == friend.id:
                raise ValueError

    def json(self):
        return {
            "id": self.id,
            "parent": self.parent,
            "body": self.body,
            "members": [p.uid for p in self.members],
            "created": str(self.created),
            "created_by": self.created_by_id
        }

class GroupComment(Base):
    __tablename__ = "group_comments"
    id = Column(Integer, primary_key=True)

    # a group comment's parent is either a group or another comment
    group_id = Column(Integer, ForeignKey("groups.id"))
    group = relationship("Group")
    parent_id = Column(Integer, ForeignKey("group_comments.id"))
    children = relationship(
        "GroupComment", backref=backref('parent', remote_side=[id]))

    body = Column(UnicodeText(length=2**31), nullable=False)

    # Accountability
    created_by_id = Column(Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship("Person")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified = Column(DateTime)  # comments may only be modified by the creator


class PersonGroupAssociation(Base):
    __tablename__ = "person_group_associations"
    person_uid = Column(Integer, ForeignKey("persons.uid"), nullable=False)
    group_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('person_uid', 'group_id', name='_uid'),)


class GroupOwnerAssociation(Base):
    __tablename__ = "group_owner_associations"
    group_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('group_id', 'owner_id', name='_uid'),)


class GroupFriendAssociation(Base):
    __tablename__ = "group_friend_associations"
    group_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('group_id', 'friend_id', name='_uid'),)


### Assets


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True)

    # null parent ID means the group belongs to the workspace.
    parent_id = Column(Integer, ForeignKey("assets.id"))
    children = relationship(
        "Asset", backref=backref('parent', remote_side=[id]))

    owners = relationship("Group", secondary="asset_owner_associations")
    friends = relationship("Group", secondary="asset_friend_associations")

    body = Column(UnicodeText(length=2**31), nullable=False)

    tasks = relationship("Task", secondary="task_asset_associations")

    # Accountability
    created_by_id = Column(
        Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship(
        "Person", primaryjoin="Person.uid == Asset.created_by_id")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified_by_id = Column(Integer, ForeignKey("persons.uid"))
    modified_by = relationship(
        "Person", primaryjoin="Person.uid == Asset.modified_by_id")
    modified = Column(DateTime)

    comments = relationship("AssetComment")


class AssetOwnerAssociation(Base):
    __tablename__ = "asset_owner_associations"
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('asset_id', 'owner_id', name='_uid'),)


class AssetFriendAssociation(Base):
    __tablename__ = "asset_friend_associations"
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('asset_id', 'friend_id', name='_uid'),)


class AssetComment(Base):
    __tablename__ = "asset_comments"
    id = Column(Integer, primary_key=True)

    # a asset comment's parent is either a asset or another comment
    asset_id = Column(Integer, ForeignKey("assets.id"))
    asset = relationship("Asset")
    parent_id = Column(Integer, ForeignKey("asset_comments.id"))
    children = relationship(
        "AssetComment", backref=backref('parent', remote_side=[id]))

    body = Column(UnicodeText(length=2**31), nullable=False)

    # Accountability
    created_by_id = Column(Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship("Person")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified = Column(DateTime)  # comments may only be modified by the creator


### Tags


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)

    # null parent ID means the group belongs to the workspace.
    parent_id = Column(Integer, ForeignKey("tags.id"))
    children = relationship(
        "Tag", backref=backref('parent', remote_side=[id]))

    owners = relationship("Group", secondary="tag_owner_associations")
    friends = relationship("Group", secondary="tag_friend_associations")

    body = Column(UnicodeText(length=2**31), nullable=False)

    # Accountability
    created_by_id = Column(
        Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship(
        "Person", primaryjoin="Person.uid == Tag.created_by_id")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified_by_id = Column(Integer, ForeignKey("persons.uid"))
    modified_by = relationship(
        "Person", primaryjoin="Person.uid == Tag.modified_by_id")
    modified = Column(DateTime)

    comments = relationship("TagComment")

    tasks = relationship("Task", secondary="task_tag_associations")


class TagOwnerAssociation(Base):
    __tablename__ = "tag_owner_associations"
    tag_id = Column(Integer, ForeignKey("tags.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('tag_id', 'owner_id', name='_uid'),)


class TagFriendAssociation(Base):
    __tablename__ = "tag_friend_associations"
    tag_id = Column(Integer, ForeignKey("tags.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('tag_id', 'friend_id', name='_uid'),)


class TagComment(Base):
    __tablename__ = "tag_comments"
    id = Column(Integer, primary_key=True)

    # a tag comment's parent is either a tag or another comment
    tag_id = Column(Integer, ForeignKey("tags.id"))
    tag = relationship("Tag")
    parent_id = Column(Integer, ForeignKey("tag_comments.id"))
    children = relationship(
        "TagComment", backref=backref('parent', remote_side=[id]))

    body = Column(UnicodeText(length=2**31), nullable=False)

    # Accountability
    created_by_id = Column(Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship("Person")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified = Column(DateTime)  # comments may only be modified by the creator


### Tasks


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)

    # null parent ID means the group belongs to the workspace.
    parent_id = Column(Integer, ForeignKey("tasks.id"))
    children = relationship(
        "Task", backref=backref('parent', remote_side=[id]))

    assigned = relationship("Group", secondary="task_assigned_associations")
    owners = relationship("Group", secondary="task_owner_associations")
    friends = relationship("Group", secondary="task_friend_associations")

    body = Column(UnicodeText(length=2**31), nullable=False)
    active = Column(Boolean, nullable=False, default=True)

    defer_until = Column(DateTime)
    due = Column(DateTime)
    estimated_duration = Column(Integer)

    started = Column(DateTime)
    completed = Column(DateTime)
    duration = Column(Integer)

    tags = relationship("Tag", secondary="task_tag_associations")
    assets = relationship("Asset", secondary="task_asset_associations")
    deps = relationship(
        "Task", secondary="task_dep_associations",
        primaryjoin="TaskDepAssociation.task_id == Task.id",
        secondaryjoin="TaskDepAssociation.dep_id == Task.id",
        backref="dependees")

    # Accountability
    created_by_id = Column(
        Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship(
        "Person", primaryjoin="Person.uid == Task.created_by_id")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified_by_id = Column(Integer, ForeignKey("persons.uid"))
    modified_by = relationship(
        "Person", primaryjoin="Person.uid == Task.modified_by_id")
    modified = Column(DateTime)

    comments = relationship("TaskComment")

    @property
    def status(self):
        if self.active:
            return "active"
        elif self.completed:
            return "complete"
        else:
            return "inactive"

    def add_subtask(self, subtask: 'Task'=None, **subtask_opts):
        if subtask is None:
            subtask = Task(**subtask_opts)

        # TODO: Check for circular dependencies
        self.children.append(subtask)
        return subtask

    def json(self):
        return {
            "id": self.id,
            "parent": self.parent_id,
            "assigned": [g.id for g in self.assigned],
            "owners": [g.id for g in self.owners],
            "friends": [g.id for g in self.friends],
            "body": self.body,
            "active": self.active,
        }


class TaskAssignedAssociation(Base):
    __tablename__ = "task_assigned_associations"
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    assigned_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('task_id', 'assigned_id', name='_uid'),)


class TaskOwnerAssociation(Base):
    __tablename__ = "task_owner_associations"
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('task_id', 'owner_id', name='_uid'),)


class TaskFriendAssociation(Base):
    __tablename__ = "task_friend_associations"
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("groups.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('task_id', 'friend_id', name='_uid'),)


class TaskAssetAssociation(Base):
    __tablename__ = "task_asset_associations"
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('task_id', 'asset_id', name='_uid'),)


class TaskTagAssociation(Base):
    __tablename__ = "task_tag_associations"
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    tag_id = Column(Integer, ForeignKey("tags.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('task_id', 'tag_id', name='_uid'),)


class TaskDepAssociation(Base):
    __tablename__ = "task_dep_associations"
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    dep_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('task_id', 'dep_id', name='_uid'),)


class TaskComment(Base):
    __tablename__ = "task_comments"
    id = Column(Integer, primary_key=True)

    # a task comment's parent is either a task or another comment
    task_id = Column(Integer, ForeignKey("tasks.id"))
    task = relationship("Task")
    parent_id = Column(Integer, ForeignKey("task_comments.id"))
    children = relationship(
        "TaskComment", backref=backref('parent', remote_side=[id]))

    body = Column(UnicodeText(length=2**31), nullable=False)

    # Accountability
    created_by_id = Column(Integer, ForeignKey("persons.uid"), nullable=False)
    created_by = relationship("Person")
    created = Column(
        DateTime, nullable=False, default=datetime.utcnow)

    modified = Column(DateTime)  # comments may only be modified by the creator
