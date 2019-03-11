"""SQL schema for FreeFocus."""
import collections
import datetime
import pathlib
import typing

import sqlalchemy as sql
from sqlalchemy import orm
from sqlalchemy.ext import declarative

from labm8 import app
from labm8 import labdate
from labm8 import sqlutil
from util.freefocus import freefocus_pb2


FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base):
  __tablename__ = 'meta'

  key = sql.Column(sql.String(255), primary_key=True)
  value = sql.Column(sql.String(255), nullable=False)


# Person.


class Person(Base):
  __tablename__ = 'persons'

  id = sql.Column(sql.Integer, primary_key=True)
  public_id = sql.Column(sql.UnicodeText(length=255), nullable=False)
  names_entries = orm.relationship('_PersonName')
  emails_entries = orm.relationship('_PersonEmail')
  groups = orm.relationship('Group', secondary='person_group_associations')

  created_at = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)
  modified_at = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  @property
  def names(self) -> typing.Iterable[str]:
    return (entry.name for entry in self.names_entries)

  @property
  def emails(self) -> typing.Iterable[str]:
    return (entry.email for entry in self.emails_entries)

  @classmethod
  def CreateFromProto(cls, session: sqlutil.Session,
                      proto: freefocus_pb2.Person) -> 'Person':
    person = sqlutil.GetOrAdd(
        session,
        cls,
        public_id=proto.id,
        created_at=labdate.DatetimeFromMillisecondsTimestamp(
            proto.created_at_utc_epoch_ms),
        modified_at=labdate.DatetimeFromMillisecondsTimestamp(
            proto.created_at_utc_epoch_ms or
            proto.most_recently_modified_at_utc_epoch_ms))
    for name in proto.name:
      sqlutil.GetOrAdd(session, _PersonName, person=person, name=name)
    for email in proto.name:
      sqlutil.GetOrAdd(session, _PersonEmail, person=person, email=email)
    return person

  def ToProto(self) -> freefocus_pb2.Person:
    workspace_groups = collections.defaultdict(list)
    for group in self.groups:
      workspace_groups[group.workspace_id].append(group.public_id)
    proto = freefocus_pb2.Person(
        id=self.public_id,
        name=list(self.names),
        email=list(self.emails),
        workspace_groups=[
            freefocus_pb2.Person.WorkspaceGroups(
                workspace_id=workspace_id,
                group_id=workspace_groups[workspace_id])
            for workspace_id in workspace_groups
        ],
        created_at_utc_epoch_ms=labdate.MillisecondsTimestamp(self.created_at),
        most_recently_modified_at_utc_epoch_ms=labdate.MillisecondsTimestamp(
            self.created_at))
    return proto


class _PersonName(Base):
  __tablename__ = 'person_names'

  person_id = sql.Column(
      sql.Integer, sql.ForeignKey('persons.id'), nullable=False)
  name = sql.Column(sql.UnicodeText(length=255), nullable=False)

  person = orm.relationship('Person')

  __table_args__ = (sql.PrimaryKeyConstraint(
      'person_id', 'name', name='person_name_key'),)


class _PersonEmail(Base):
  __tablename__ = 'person_emails'

  person_id = sql.Column(
      sql.Integer, sql.ForeignKey('persons.id'), nullable=False)
  email = sql.Column(sql.String(512), nullable=False)

  person = orm.relationship('Person')

  __table_args__ = (sql.PrimaryKeyConstraint(
      'person_id', 'email', name='person_email_key'),)


# Workspace


class Workspace(Base):
  __tablename__ = 'workspaces'

  id = sql.Column(sql.Integer, primary_key=True)
  public_id = sql.Column(sql.UnicodeText(length=255), nullable=False)

  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  owners = orm.relationship(
      'Group',
      secondary='workspace_owner_associations',
      primaryjoin='WorkspaceOwnerAssociation.workspace_id == Group.id',
      secondaryjoin='WorkspaceOwnerAssociation.owner_id == Group.id')

  friends = orm.relationship(
      'Group',
      secondary='workspace_friend_associations',
      primaryjoin='WorkspaceFriendAssociation.workspace_id == Group.id',
      secondaryjoin='WorkspaceFriendAssociation.friend_id == Group.id')

  comments = orm.relationship('WorkspaceComment')


class WorkspaceOwnerAssociation(Base):
  __tablename__ = 'workspace_owner_associations'
  workspace_id = sql.Column(
      sql.Integer, sql.ForeignKey('workspaces.id'), nullable=False)
  owner_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'workspace_id', 'owner_id', name='_id'),)


class WorkspaceFriendAssociation(Base):
  __tablename__ = 'workspace_friend_associations'
  workspace_id = sql.Column(
      sql.Integer, sql.ForeignKey('workspaces.id'), nullable=False)
  friend_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'workspace_id', 'friend_id', name='_id'),)


class WorkspaceComment(Base):
  __tablename__ = 'workspace_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a workspace comment's parent is either a workspace or another comment
  workspace_id = sql.Column(sql.Integer, sql.ForeignKey('workspaces.id'))
  workspace = orm.relationship('Workspace')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('workspace_comments.id'))
  children = orm.relationship(
      'WorkspaceComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship('Group')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  modified = sql.Column(
      sql.DateTime)  # comments may only be modified by the creator


### Groups


class Group(Base):
  __tablename__ = 'groups'

  id = sql.Column(sql.Integer, primary_key=True)

  # null parent ID means the group belongs to the workspace.
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('groups.id'))
  children = orm.relationship(
      'Group',
      primaryjoin='Group.parent_id == Group.id',
      backref=orm.backref('parent', remote_side=[id]))

  owners = orm.relationship(
      'Group',
      secondary='group_owner_associations',
      primaryjoin='GroupOwnerAssociation.group_id == Group.id',
      secondaryjoin='GroupOwnerAssociation.owner_id == Group.id')

  friends = orm.relationship(
      'Group',
      secondary='group_friend_associations',
      primaryjoin='GroupFriendAssociation.group_id == Group.id',
      secondaryjoin='GroupFriendAssociation.friend_id == Group.id')

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  members = orm.relationship('Person', secondary='person_group_associations')

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=True)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == Group.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  modified_by_id = sql.Column(sql.Integer, sql.ForeignKey('persons.id'))
  modified_by = orm.relationship(
      'Person', primaryjoin='Person.id == Group.modified_by_id')
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
        'members': [p.id for p in self.members],
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

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == GroupComment.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  modified = sql.Column(
      sql.DateTime)  # comments may only be modified by the creator


class PersonGroupAssociation(Base):
  __tablename__ = 'person_group_associations'
  person_id = sql.Column(
      sql.Integer, sql.ForeignKey('persons.id'), nullable=False)
  group_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'person_id', 'group_id', name='person_group_key'),)


class GroupOwnerAssociation(Base):
  __tablename__ = 'group_owner_associations'
  group_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  owner_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'group_id', 'owner_id', name='group_owner_key'),)


class GroupFriendAssociation(Base):
  __tablename__ = 'group_friend_associations'
  group_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  friend_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'group_id', 'friend_id', name='_id'),)


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

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  tasks = orm.relationship('Task', secondary='task_asset_associations')

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == Asset.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  modified_by_id = sql.Column(sql.Integer, sql.ForeignKey('persons.id'))
  modified_by = orm.relationship(
      'Person', primaryjoin='Person.id == Asset.modified_by_id')
  modified = sql.Column(sql.DateTime)

  comments = orm.relationship('AssetComment')


class AssetOwnerAssociation(Base):
  __tablename__ = 'asset_owner_associations'
  asset_id = sql.Column(
      sql.Integer, sql.ForeignKey('assets.id'), nullable=False)
  owner_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'asset_id', 'owner_id', name='_id'),)


class AssetFriendAssociation(Base):
  __tablename__ = 'asset_friend_associations'
  asset_id = sql.Column(
      sql.Integer, sql.ForeignKey('assets.id'), nullable=False)
  friend_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'asset_id', 'friend_id', name='_id'),)


class AssetComment(Base):
  __tablename__ = 'asset_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a asset comment's parent is either a asset or another comment
  asset_id = sql.Column(sql.Integer, sql.ForeignKey('assets.id'))
  asset = orm.relationship('Asset')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('asset_comments.id'))
  children = orm.relationship(
      'AssetComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship('Group')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

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

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == Tag.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  modified_by_id = sql.Column(sql.Integer, sql.ForeignKey('persons.id'))
  modified_by = orm.relationship(
      'Person', primaryjoin='Person.id == Tag.modified_by_id')
  modified = sql.Column(sql.DateTime)

  comments = orm.relationship('TagComment')

  tasks = orm.relationship('Task', secondary='task_tag_associations')


class TagOwnerAssociation(Base):
  __tablename__ = 'tag_owner_associations'
  tag_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'), nullable=False)
  owner_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint('tag_id', 'owner_id', name='_id'),)


class TagFriendAssociation(Base):
  __tablename__ = 'tag_friend_associations'
  tag_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'), nullable=False)
  friend_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint('tag_id', 'friend_id',
                                             name='_id'),)


class TagComment(Base):
  __tablename__ = 'tag_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a tag comment's parent is either a tag or another comment
  tag_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'))
  tag = orm.relationship('Tag')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('tag_comments.id'))
  children = orm.relationship(
      'TagComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship('Group')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

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

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

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
      'Task',
      secondary='task_dep_associations',
      primaryjoin='TaskDepAssociation.task_id == Task.id',
      secondaryjoin='TaskDepAssociation.dep_id == Task.id',
      backref='dependees')

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship(
      'Group', primaryjoin='Group.id == Task.created_by_id')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  modified_by_id = sql.Column(sql.Integer, sql.ForeignKey('persons.id'))
  modified_by = orm.relationship(
      'Person', primaryjoin='Person.id == Task.modified_by_id')
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
    return self.defer_until and self.defer_until > datetime.datetime.utcnow()

  @property
  def is_assigned(self):
    return len(self.assigned) > 0

  @property
  def is_overdue(self):
    return self.due and self.due > datetime.datetime.utcnow()

  def add_subtask(self, subtask: 'Task' = None, **subtask_opts):
    if subtask is None:
      subtask = Task(**subtask_opts)

    # TODO: Check for circular dependencies
    self.children.append(subtask)
    return subtask


class TaskAssignedAssociation(Base):
  __tablename__ = 'task_assigned_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  assigned_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'task_id', 'assigned_id', name='_id'),)


class TaskOwnerAssociation(Base):
  __tablename__ = 'task_owner_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  owner_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint('task_id', 'owner_id',
                                             name='_id'),)


class TaskFriendAssociation(Base):
  __tablename__ = 'task_friend_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  friend_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint(
      'task_id', 'friend_id', name='_id'),)


class TaskAssetAssociation(Base):
  __tablename__ = 'task_asset_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  asset_id = sql.Column(
      sql.Integer, sql.ForeignKey('assets.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint('task_id', 'asset_id',
                                             name='_id'),)


class TaskTagAssociation(Base):
  __tablename__ = 'task_tag_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  tag_id = sql.Column(sql.Integer, sql.ForeignKey('tags.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint('task_id', 'tag_id', name='_id'),)


class TaskDepAssociation(Base):
  __tablename__ = 'task_dep_associations'
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  dep_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'), nullable=False)
  __table_args__ = (sql.PrimaryKeyConstraint('task_id', 'dep_id', name='_id'),)


class TaskComment(Base):
  __tablename__ = 'task_comments'
  id = sql.Column(sql.Integer, primary_key=True)

  # a task comment's parent is either a task or another comment
  task_id = sql.Column(sql.Integer, sql.ForeignKey('tasks.id'))
  task = orm.relationship('Task')
  parent_id = sql.Column(sql.Integer, sql.ForeignKey('task_comments.id'))
  children = orm.relationship(
      'TaskComment', backref=orm.backref('parent', remote_side=[id]))

  body = sql.Column(sql.UnicodeText(length=2**31), nullable=False)

  # Accountability
  created_by_id = sql.Column(
      sql.Integer, sql.ForeignKey('groups.id'), nullable=False)
  created_by = orm.relationship('Group')
  created = sql.Column(
      sql.DateTime, nullable=False, default=datetime.datetime.utcnow)

  modified = sql.Column(
      sql.DateTime)  # comments may only be modified by the creator


class Database(sqlutil.Database):
  """The FreeFocus database."""

  def __init__(self, path: pathlib.Path):
    super(Database, self).__init__(f'sqlite:///{path}', Base)
