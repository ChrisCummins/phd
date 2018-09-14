"""SQL schema for FreeFocus."""
import pathlib
import sqlalchemy as sql
from absl import flags
from datetime import datetime
from sqlalchemy import orm
from sqlalchemy.ext import declarative

from lib.labm8 import sqlutil


FLAGS = flags.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  __tablename__ = 'meta'

  key = sql.Column(sql.String(255), primary_key=True)
  value = sql.Column(sql.String(255), nullable=False)


# Person.


class Person(Base):
  __tablename__ = 'persons'

  uid = sql.Column(sql.Integer, primary_key=True)
  name = sql.Column(sql.UnicodeText(length=255), nullable=False)

  emails = orm.relationship('Email')
  groups = orm.relationship('Group', secondary='person_group_associations')

  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  def json(self):
    return {
      'uid': self.uid,
      'name': self.name,
      'created': str(self.created),
    }


class Email(Base):
  __tablename__ = 'email_addresses'

  person_uid = sql.Column(sql.Integer, sql.ForeignKey('persons.uid'),
                          nullable=False)
  person = orm.relationship('Person')
  address = sql.Column(sql.String(255), nullable=False)

  __table_args__ = (
    sql.PrimaryKeyConstraint('person_uid', 'address', name='_uid'),)


### Workspace


class Workspace(Base):
  """ only one Workspace per database """
  __tablename__ = 'workspaces'

  uid = sql.Column(sql.String(255), primary_key=True)
  created = sql.Column(sql.DateTime, nullable=False, default=datetime.utcnow)

  owners = orm.relationship(
      'Group', secondary='workspace_owner_associations',
      primaryjoin='WorkspaceOwnerAssociation.workspace_uid == Group.id',
      secondaryjoin='WorkspaceOwnerAssociation.owner_id == Group.id')

  friends = orm.relationship(
      'Group', secondary='workspace_friend_associations',
      primaryjoin='WorkspaceFriendAssociation.workspace_uid == Group.id',
      secondaryjoin='WorkspaceFriendAssociation.friend_id == Group.id')

  comments = orm.relationship('WorkspaceComment')


class WorkspaceOwnerAssociation(Base):
  __tablename__ = 'workspace_owner_associations'
  workspace_uid = sql.Column(sql.Integer, sql.ForeignKey('workspaces.uid'),
                             nullable=False)
  owner_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                        nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('workspace_uid', 'owner_id', name='_uid'),)


class WorkspaceFriendAssociation(Base):
  __tablename__ = 'workspace_friend_associations'
  workspace_uid = sql.Column(sql.Integer, sql.ForeignKey('workspaces.uid'),
                             nullable=False)
  friend_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                         nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('workspace_uid', 'friend_id', name='_uid'),)


class WorkspaceComment(Base):
  __tablename__ = 'workspace_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a workspace comment's parent is either a workspace or another comment
  workspace_uid = sql.Column(sql.Integer, sql.ForeignKey('workspaces.uid'))
  workspace = orm.relationship('Workspace')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('workspace_comments.id'))
  children = orm.relationship(
      'WorkspaceComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  # Accountability
  created_by_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                             nullable=False)
  created_by = orm.relationship('Group')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified = sql.Column(
      sql.DateTime)  # comments may only be modified by the creator


### Groups


class Group(Base):
  __tablename__ = 'groups'

  id = sql.Column(sql.Integer, primary_key=True)

  # null parent ID means the group belongs to the workspace.
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'))
  children = orm.relationship(
      'Group', primaryjoin='Group.parent_id == Group.id',
      backref=orm.backref('parent', remote_side=[id]))

  owners = orm.relationship(
      'Group', secondary='group_owner_associations',
      primaryjoin='GroupOwnerAssociation.group_id == Group.id',
      secondaryjoin='GroupOwnerAssociation.owner_id == Group.id')

  friends = orm.relationship(
      'Group', secondary='group_friend_associations',
      primaryjoin='GroupFriendAssociation.group_id == Group.id',
      secondaryjoin='GroupFriendAssociation.friend_id == Group.id')

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  members = orm.relationship('Person', secondary='person_group_associations')

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=True)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == Group.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified_by_id = sql.Column(sql.Integer, sql.ForeignKey('persons.uid'))
  modified_by = orm.relationship(
      'Person', primaryjoin='Person.uid == Group.modified_by_id')
  modified = sql.Column(sql.DateTime)

  comments = orm.relationship(
      'GroupComment', primaryjoin='GroupComment.group_id == Group.id')

  def validate(self):
    for owner in self.owners:
      if self.id == owner.id:
        raise ValueError

    for friend in self.friends:
      if self.id == friend.id:
        raise ValueError

  def json(self):
    return {
      'id': self.id,
      'parent': self.parent,
      'body': self.body,
      'members': [p.uid for p in self.members],
      'created': str(self.created),
      'created_by': self.created_by_id
    }


class GroupComment(Base):
  __tablename__ = 'group_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a group comment's parent is either a group or another comment
  group_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'))
  group = orm.relationship(
      'Group', primaryjoin='Group.id == GroupComment.group_id')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('group_comments.id'))
  children = orm.relationship(
      'GroupComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  # Accountability
  created_by_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                             nullable=False)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == GroupComment.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified = sql.Column(
      sql.DateTime)  # comments may only be modified by the creator


class PersonGroupAssociation(Base):
  __tablename__ = 'person_group_associations'
  person_uid = sql.Column(sql.Integer, sql.ForeignKey('persons.uid'),
                          nullable=False)
  group_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                        nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('person_uid', 'group_id', name='_uid'),)


class GroupOwnerAssociation(Base):
  __tablename__ = 'group_owner_associations'
  group_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                        nullable=False)
  owner_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                        nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('group_id', 'owner_id', name='_uid'),)


class GroupFriendAssociation(Base):
  __tablename__ = 'group_friend_associations'
  group_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                        nullable=False)
  friend_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                         nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('group_id', 'friend_id', name='_uid'),)


### Assets


class Asset(Base):
  __tablename__ = 'assets'

  id = sql.Column(sql.Integer, primary_key=True)

  # null parent ID means the group belongs to the workspace.
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('assets.id'))
  children = orm.relationship(
      'Asset', backref=orm.backref('parent', remote_side=[id]))

  owners = orm.relationship('Group', secondary='asset_owner_associations')
  friends = orm.relationship('Group', secondary='asset_friend_associations')

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  tasks = orm.relationship('Task', secondary='task_asset_associations')

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == Asset.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified_by_id = sql.Column(sql.Integer, sql.ForeignKey('persons.uid'))
  modified_by = orm.relationship(
      'Person', primaryjoin='Person.uid == Asset.modified_by_id')
  modified = sql.Column(sql.DateTime)

  comments = orm.relationship('AssetComment')


class AssetOwnerAssociation(Base):
  __tablename__ = 'asset_owner_associations'
  asset_id = sql.Column(sql.Integer, sql.ForeignKey('assets.id'),
                        nullable=False)
  owner_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                        nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('asset_id', 'owner_id', name='_uid'),)


class AssetFriendAssociation(Base):
  __tablename__ = 'asset_friend_associations'
  asset_id = sql.Column(sql.Integer, sql.ForeignKey('assets.id'),
                        nullable=False)
  friend_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                         nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('asset_id', 'friend_id', name='_uid'),)


class AssetComment(Base):
  __tablename__ = 'asset_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a asset comment's parent is either a asset or another comment
  asset_id = sql.Column(sql.Integer, sql.ForeignKey('assets.id'))
  asset = orm.relationship('Asset')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('asset_comments.id'))
  children = orm.relationship(
      'AssetComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  # Accountability
  created_by_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                             nullable=False)
  created_by = orm.relationship('Group')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified = sql.Column(
      sql.DateTime)  # comments may only be modified by the creator


### Tags


class Tag(Base):
  __tablename__ = 'tags'

  id = sql.Column(sql.Integer, primary_key=True)

  # null parent ID means the group belongs to the workspace.
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'))
  children = orm.relationship(
      'Tag', backref=orm.backref('parent', remote_side=[id]))

  owners = orm.relationship('Group', secondary='tag_owner_associations')
  friends = orm.relationship('Group', secondary='tag_friend_associations')

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == Tag.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified_by_id = sql.Column(sql.Integer, sql.ForeignKey('persons.uid'))
  modified_by = orm.relationship(
      'Person', primaryjoin='Person.uid == Tag.modified_by_id')
  modified = sql.Column(sql.DateTime)

  comments = orm.relationship('TagComment')

  tasks = orm.relationship('Task', secondary='task_tag_associations')


class TagOwnerAssociation(Base):
  __tablename__ = 'tag_owner_associations'
  tag_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'), nullable=False)
  owner_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                        nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('tag_id', 'owner_id', name='_uid'),)


class TagFriendAssociation(Base):
  __tablename__ = 'tag_friend_associations'
  tag_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'), nullable=False)
  friend_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                         nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('tag_id', 'friend_id', name='_uid'),)


class TagComment(Base):
  __tablename__ = 'tag_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a tag comment's parent is either a tag or another comment
  tag_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'))
  tag = orm.relationship('Tag')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('tag_comments.id'))
  children = orm.relationship(
      'TagComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  # Accountability
  created_by_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                             nullable=False)
  created_by = orm.relationship('Group')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified = sql.Column(
      sql.DateTime)  # comments may only be modified by the creator


### Tasks


class Task(Base):
  __tablename__ = 'tasks'

  id = sql.Column(sql.Integer, primary_key=True)

  # null parent ID means the group belongs to the workspace.
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'))
  children = orm.relationship(
      'Task', backref=orm.backref('parent', remote_side=[id]))

  assigned = orm.relationship('Group', secondary='task_assigned_associations')
  owners = orm.relationship('Group', secondary='task_owner_associations')
  friends = orm.relationship('Group', secondary='task_friend_associations')

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  defer_until = sql.Column(sql.DateTime)
  due = sql.Column(sql.DateTime)
  estimated_duration = sql.Column(sql.Integer)
  start_on = sql.Column(sql.DateTime)
  started = sql.Column(sql.DateTime)
  completed = sql.Column(sql.DateTime)
  duration = sql.Column(sql.Integer)

  tags = orm.relationship('Tag', secondary='task_tag_associations')
  assets = orm.relationship('Asset', secondary='task_asset_associations')
  deps = orm.relationship(
      'Task', secondary='task_dep_associations',
      primaryjoin='TaskDepAssociation.task_id == Task.id',
      secondaryjoin='TaskDepAssociation.dep_id == Task.id',
      backref='dependees')

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == Task.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified_by_id = sql.Column(sql.Integer, sql.ForeignKey('persons.uid'))
  modified_by = orm.relationship(
      'Person', primaryjoin='Person.uid == Task.modified_by_id')
  modified = sql.Column(sql.DateTime)

  comments = orm.relationship('TaskComment')

  @property
  def status(self):
    if self.is_deferred:
      return 'deferred'
    elif self.completed:
      return 'complete'
    elif self.is_blocked:
      return 'blocked'
    elif self.is_overdue:
      return 'overdue'
    elif self.started:
      return 'started'
    else:
      return 'active'

  @property
  def is_blocked(self):
    return len(self.deps) > 0

  @property
  def is_deferred(self):
    return self.defer_until and self.defer_until > datetime.utcnow()

  @property
  def is_assigned(self):
    return len(self.assigned) > 0

  @property
  def is_overdue(self):
    return self.due and self.due > datetime.utcnow()

  def add_subtask(self, subtask: 'Task' = None, **subtask_opts):
    if subtask is None:
      subtask = Task(**subtask_opts)

    # TODO: Check for circular dependencies
    self.children.append(subtask)
    return subtask


class TaskAssignedAssociation(Base):
  __tablename__ = 'task_assigned_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  assigned_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                           nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('task_id', 'assigned_id', name='_uid'),)


class TaskOwnerAssociation(Base):
  __tablename__ = 'task_owner_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  owner_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                        nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('task_id', 'owner_id', name='_uid'),)


class TaskFriendAssociation(Base):
  __tablename__ = 'task_friend_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  friend_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                         nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('task_id', 'friend_id', name='_uid'),)


class TaskAssetAssociation(Base):
  __tablename__ = 'task_asset_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  asset_id = sql.Column(sql.Integer, sql.ForeignKey('assets.id'),
                        nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('task_id', 'asset_id', name='_uid'),)


class TaskTagAssociation(Base):
  __tablename__ = 'task_tag_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  tag_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'), nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('task_id', 'tag_id', name='_uid'),)


class TaskDepAssociation(Base):
  __tablename__ = 'task_dep_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  dep_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  __table_args__ = (
    sql.PrimaryKeyConstraint('task_id', 'dep_id', name='_uid'),)


class TaskComment(Base):
  __tablename__ = 'task_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a task comment's parent is either a task or another comment
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'))
  task = orm.relationship('Task')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('task_comments.id'))
  children = orm.relationship(
      'TaskComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2 ** 31), nullable=False)

  # Accountability
  created_by_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'),
                             nullable=False)
  created_by = orm.relationship('Group')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.utcnow)

  modified = sql.Column(
      sql.DateTime)  # comments may only be modified by the creator


class Database(sqlutil.Database):
  """The FreeFocus database."""

  def __init__(self, path: pathlib.Path):
    super(Database, self).__init__(path, Base)
