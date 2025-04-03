from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Any


class Permission(Enum):
    """
    Fine-grained permissions for model analysis operations.

    These permissions control what operations the model inspector is allowed
    to perform when analyzing models, providing security control.
    """
    READ_FILE = auto()  # Permission to read files
    READ_METADATA = auto()  # Permission to read metadata from files
    READ_WEIGHTS = auto()  # Permission to read model weights
    EXECUTE_CODE = auto()  # Permission to execute code (risky)
    NETWORK_ACCESS = auto()  # Permission to access the network
    CREATE_SUBPROCESS = auto()  # Permission to create subprocesses
    WRITE_FILE = auto()  # Permission to write files
    USE_GPU = auto()  # Permission to use GPU acceleration

    @classmethod
    def safe_set(cls) -> Set['Permission']:
        """
        Get the set of permissions considered safe.

        Returns:
            Set of safe permissions
        """
        return {cls.READ_FILE, cls.READ_METADATA}

    @classmethod
    def default_set(cls) -> Set['Permission']:
        """
        Get the default set of permissions.

        Returns:
            Set of default permissions
        """
        return {cls.READ_FILE, cls.READ_METADATA, cls.READ_WEIGHTS, cls.USE_GPU}

    @classmethod
    def all_permissions(cls) -> Set['Permission']:
        """
        Get all available permissions.

        Returns:
            Set of all permissions
        """
        return set(cls)


@dataclass
class PermissionSet:
    """
    A set of permissions with format-specific overrides.

    This class manages a base set of permissions that apply to all formats,
    plus format-specific permission overrides.
    """
    base_permissions: Set[Permission] = field(default_factory=Permission.default_set)
    format_permissions: Dict[str, Set[Permission]] = field(default_factory=dict)

    def format_has_permission(self, format_name: str, permission: Permission) -> bool:
        """
        Check if a specific format has a specific permission.

        Args:
            format_name: The file format (e.g., '.safetensors')
            permission: The permission to check

        Returns:
            True if the format has the permission, False otherwise
        """
        # Normalize format name
        if not format_name.startswith('.'):
            format_name = f'.{format_name}'
        format_name = format_name.lower()

        # Check format-specific permissions if they exist
        if format_name in self.format_permissions:
            return permission in self.format_permissions[format_name]

        # Otherwise check base permissions
        return permission in self.base_permissions

    def has_permission(self, permission: Permission) -> bool:
        """
        Check if the base permission set includes a specific permission.

        Args:
            permission: The permission to check

        Returns:
            True if the permission is granted, False otherwise
        """
        return permission in self.base_permissions

    def set_permission(self, permission: Permission, granted: bool = True) -> None:
        """
        Set or revoke a base permission.

        Args:
            permission: The permission to set
            granted: True to grant the permission, False to revoke it
        """
        if granted:
            self.base_permissions.add(permission)
        else:
            self.base_permissions.discard(permission)

    def set_format_permission(
            self,
            format_name: str,
            permission: Permission,
            granted: bool = True
    ) -> None:
        """
        Set or revoke a permission for a specific format.

        Args:
            format_name: The file format (e.g., '.safetensors')
            permission: The permission to set
            granted: True to grant the permission, False to revoke it
        """
        # Normalize format name
        if not format_name.startswith('.'):
            format_name = f'.{format_name}'
        format_name = format_name.lower()

        # Initialize format permissions if needed
        if format_name not in self.format_permissions:
            # Start with base permissions
            self.format_permissions[format_name] = self.base_permissions.copy()

        # Grant or revoke the permission
        if granted:
            self.format_permissions[format_name].add(permission)
        else:
            self.format_permissions[format_name].discard(permission)

    @classmethod
    def create_safe_permissions(cls) -> 'PermissionSet':
        """
        Create a permission set with only safe permissions.

        Returns:
            PermissionSet with only safe permissions
        """
        return cls(base_permissions=Permission.safe_set())

    @classmethod
    def create_unrestricted_permissions(cls) -> 'PermissionSet':
        """
        Create a permission set with all permissions granted.

        Returns:
            PermissionSet with all permissions
        """
        return cls(base_permissions=Permission.all_permissions())
