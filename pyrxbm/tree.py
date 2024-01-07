from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .datatypes import UniqueId


# fmt: off
class PropertyType(Enum):
    Unknown            =  0
    String             =  1
    Bool               =  2
    Int                =  3
    Float              =  4
    Double             =  5
    UDim               =  6
    UDim2              =  7
    Ray                =  8
    Faces              =  9
    Axes               = 10
    BrickColor         = 11
    Color3             = 12
    Vector2            = 13
    Vector3            = 14

    CFrame             = 16
    Quaternion         = 17
    Enum               = 18
    Ref                = 19
    Vector3int16       = 20
    NumberSequence     = 21
    ColorSequence      = 22
    NumberRange        = 23
    Rect               = 24
    PhysicalProperties = 25
    Color3uint8        = 26
    Int64              = 27
    SharedString       = 28
    ProtectedString    = 29
    OptionalCFrame     = 30
    UniqueId           = 31
    FontFace           = 32
# fmt: on


@dataclass
class Instance:
    """
    Describes an object in Roblox's DataModel hierarchy.
    Instances can have sets of properties loaded from *.rbxl/*.rbxm files.
    """

    """ The name of this Instance. """
    Name: str = ""
    """ The source AssetId this instance was created in. """
    SourceAssetId: int = field(default=-1, repr=False)
    """ A hashset of CollectionService tags assigned to this Instance. """
    Tags: bytes = field(default=b"", repr=False)
    """ The public readonly access point of the attributes on this Instance. """
    AttributesSerialize: bytes = field(default=b"", repr=False)

    def __post_init__(self):
        self.Name = self.Name or self.ClassName
        """ Indicates whether this Instance should be serialized. """
        self.Archivable: bool = True  # not a field because not serialized

        """ The raw list of children for this Instance. """
        self._children: list[Instance] = []
        """ The raw unsafe value of the Instance's parent. """
        self._parent: Instance = None
        """ A unique identifier declared for this instance. """
        self.UniqueId: UniqueId = None
        """ A context-dependent unique identifier for this instance when being serialized. """
        self.referent: str = None
        """ Indicates whether the parent of this object is locked. """
        self._parent_locked: bool = False
        """ Indicates whether this Instance is a Service. """
        self.is_service: bool = False
        """ Indicates whether this Instance has been destroyed. """
        self.Destroyed: bool = False
        """ The public readonly access point of the attributes on this Instance. """
        # self.Attributes: RbxAttributes = RbxAttributes()
        # self.RefreshProperties()

    @property
    def ClassName(self):
        """The ClassName of this Instance."""
        return self.__class__.__name__

    # @property
    # def AttributesSerialize(self) -> bytes:
    #     """The internal serialized data of this Instance's attributes"""
    #     return self.Attributes.Save()
    #
    # @AttributesSerialize.setter
    # def AttributesSerialize(self, value: bytes):
    #     self.Attributes.Load(value)
    #
    # @property
    # def SerializedTags(self) -> bytes:
    #     """
    #     Internal format of the Instance's CollectionService tags.
    #     Property objects will look to this member for serializing the Tags property.
    #     """
    #     if not self.Tags:
    #         return None
    #     return "\0".join(self.Tags).encode()
    #
    # @SerializedTags.setter
    # def SerializedTags(self, value: bytes):
    #     """
    #     Internal format of the Instance's CollectionService tags.
    #     Property objects will look to this member for serializing the Tags property.
    #     """
    #     buffer = bytearray()
    #     self.Tags.clear()
    #     for i, id in enumerate(value):
    #         if id != 0:
    #             buffer.append(id)
    #         if id == 0 or i == (len(value) - 1):
    #             self.Tags.append(buffer.decode())
    #             buffer.clear()

    """
        /// <summary>
    /// Attempts to get the value of an attribute whose type is T.
    /// Returns false if no attribute was found with that type.
        /// </summary>
    /// <typeparam name="T">The expected type of the attribute.</typeparam>
    /// <param name="key">The name of the attribute.</param>
    /// <param name="value">The out value to set.</param>
    /// <returns>True if the attribute could be read and the out value was set, false otherwise.</returns>
    public bool GetAttribute<T>(string key, out T value)
    {
        if (Attributes.TryGetValue(key, out RbxAttribute attr))
        {
            if (attr?.Value is T result)
            {
                value = result;
                return true;
            }
        }

        value = default;
        return false;
    }

        /// <summary>
    /// Attempts to set an attribute to the provided value. The provided key must be no longer than 100 characters.
    /// Returns false if the key is too long or the provided type is not supported by Roblox.
    /// If an attribute with the provide key already exists, it will be overwritten.
        /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="key">The name of the attribute.</param>
    /// <param name="value">The value to be assigned to the attribute.</param>
    /// <returns>True if the attribute was set, false otherwise.</returns>
    public bool SetAttribute<T>(string key, T value)
    {
        if (key.Length > 100)
            return false;

        if (key.StartsWith("RBX", StringComparison.InvariantCulture))
            return false;

        if (value == null)
        {
            Attributes[key] = null;
            return true;
        }

        Type type = value.GetType();

        if (!RbxAttribute.SupportsType(type))
            return false;

        var attr = new RbxAttribute(value);
        Attributes[key] = attr;

        return true;
    }
    """

    def IsAncestorOf(self, descendant: Instance | None) -> bool:
        """
        :param descendant: The instance whose descendance will be tested against this Instance.
        :return: True if this Instance is an ancestor to the provided Instance.
        """
        while descendant is not None:
            if descendant == self:
                return True
            descendant = descendant.Parent

        return False

    def IsDescendantOf(self, ancestor: Instance | None) -> bool:
        """
        :param ancestor: The instance whose ancestry will be tested against this Instance.
        :return: True if this Instance is a descendant of the provided Instance.
        """
        return ancestor.IsAncestorOf(self) if ancestor is not None else False

    """
    /// <summary>
/// Returns true if the provided instance inherits from the provided instance type.
    /// </summary>
[Obsolete("Use the `is` operator instead.")]
public bool IsA<T>() where T : Instance
{
    return this is T;
}

    /// <summary>
/// Attempts to cast this Instance to an inherited class of type '<typeparamref name="T"/>'.
/// Returns null if the instance cannot be casted to the provided type.
    /// </summary>
/// <typeparam name="T">The type of Instance to cast to.</typeparam>
/// <returns>The instance as the type '<typeparamref name="T"/>' if it can be converted, or null.</returns>
[Obsolete("Use the `as` operator instead.")]
public T Cast<T>() where T : Instance
{
    return this as T;
}
"""

    @property
    def Children(self) -> list[Instance]:
        """The raw list of children for this Instance."""
        return self._children[:]

    @property
    def Parent(self) -> Instance | None:
        """
        The parent of this Instance, or null if the instance is the root of a tree.<para/>
        Setting the value of this property will throw an exception if:<para/>
        - The parent is currently locked.<para/>
        - The value is set to itself.<para/>
        - The value is a descendant of the Instance.
        """

        return self._parent

    @Parent.setter
    def Parent(self, value: Instance | None):
        if self._parent_locked:
            new_parent = value.Name if value is not None else "NULL"
            curr_parent = self.Parent.Name if self.Parent is not None else "NULL"

            raise ValueError(
                f"The Parent property of {self.Name} is locked, current parent: {curr_parent}, new parent {new_parent}"
            )

        if self.IsAncestorOf(value):
            path_a = self.GetFullName(".")
            path_b = value.GetFullName(".")
            raise ValueError(
                f"Attempt to set parent of {path_a} to {path_b} would result in a circular reference"
            )

        if self.Parent == self:
            raise ValueError(f"Attempt to set {self.Name} as its own parent")

        if self._parent is not None:
            self._parent._children.remove(self)
        if value is not None:
            value._children.append(self)
        self._parent = value

    """

    /// <summary>
/// Returns an array containing all the children of this Instance.
    /// </summary>
public Instance[] GetChildren()
{
    return _children.ToArray();
}

    /// <summary>
/// Returns an array containing all the children of this Instance, whose type is '<typeparamref name="T"/>'.
    /// </summary>
public T[] GetChildrenOfType<T>() where T : Instance
{
    T[] ofType = GetChildren()
        .Where(child => child is T)
        .Cast<T>()
        .ToArray();

    return ofType;
}

    /// <summary>
/// Returns an array containing all the descendants of this Instance.
    /// </summary>
public Instance[] GetDescendants()
{
    var results = new List<Instance>();

    foreach (Instance child in _children)
    {
        // Add this child to the results.
        results.Add(child);

        // Add its descendants to the results.
        Instance[] descendants = child.GetDescendants();
        results.AddRange(descendants);
    }

    return results.ToArray();
}

    /// <summary>
/// Returns an array containing all the descendants of this Instance, whose type is '<typeparamref name="T"/>'.
    /// </summary>
public T[] GetDescendantsOfType<T>() where T : Instance
{
    T[] ofType = GetDescendants()
        .Where(desc => desc is T)
        .Cast<T>()
        .ToArray();

    return ofType;
}

    /// <summary>
/// Returns the first child of this Instance whose Name is the provided string name.
/// If the instance is not found, this returns null.
    /// </summary>
/// <param name="name">The Name of the Instance to find.</param>
/// <param name="recursive">Indicates if we should search descendants as well.</param>
public T FindFirstChild<T>(string name, bool recursive = false) where T : Instance
{
    T result = null;

    var query = _children
        .Where(child => child is T)
        .Where(child => name == child.Name)
        .Cast<T>();

    if (query.Any())
    {
        result = query.First();
    }
    else if (recursive)
    {
        foreach (Instance child in _children)
        {
            T found = child.FindFirstChild<T>(name, true);

            if (found != null)
            {
                result = found;
                break;
            }
        }
    }

    return result;
}

    /// <summary>
/// Returns the first child of this Instance whose Name is the provided string name.
/// If the instance is not found, this returns null.
    /// </summary>
/// <param name="name">The Name of the Instance to find.</param>
/// <param name="recursive">Indicates if we should search descendants as well.</param>
public Instance FindFirstChild(string name, bool recursive = false)
{
    return FindFirstChild<Instance>(name, recursive);
}

    /// <summary>
/// Returns the first ancestor of this Instance whose Name is the provided string name.
/// If the instance is not found, this returns null.
    /// </summary>
/// <param name="name">The Name of the Instance to find.</param>
public T FindFirstAncestor<T>(string name) where T : Instance
{
    Instance ancestor = Parent;

    while (ancestor != null)
    {
        if (ancestor is T && ancestor.Name == name)
            return ancestor as T;
        
        ancestor = ancestor.Parent;
    }

    return null;
}

    /// <summary>
/// Returns the first ancestor of this Instance whose Name is the provided string name.
/// If the instance is not found, this returns null.
    /// </summary>
/// <param name="name">The Name of the Instance to find.</param>
public Instance FindFirstAncestor(string name)
{
    return FindFirstAncestor<Instance>(name);
}

    /// <summary>
/// Returns the first ancestor of this Instance whose ClassName is the provided string className.
/// If the instance is not found, this returns null.
    /// </summary>
/// <param name="name">The Name of the Instance to find.</param>
public T FindFirstAncestorOfClass<T>() where T : Instance
{
    Instance ancestor = Parent;

    while (ancestor != null)
    {
        if (ancestor is T)
            return ancestor as T;
        
        ancestor = ancestor.Parent;
    }

    return null;
}

    /// <summary>
/// Returns the first ancestor of this Instance which derives from the provided type <typeparamref name="T"/>.
/// If the instance is not found, this returns null.
    /// </summary>
/// <param name="name">The Name of the Instance to find.</param>
public T FindFirstAncestorWhichIsA<T>() where T : Instance
{
    T ancestor = null;
    Instance check = Parent;

    while (check != null)
    {
        if (check is T)
        {
            ancestor = check as T;
            break;
        }

        check = check.Parent;
    }

    return ancestor;
}

    /// <summary>
/// Returns the first Instance whose ClassName is the provided string className.
/// If the instance is not found, this returns null.
    /// </summary>
/// <param name="className">The ClassName of the Instance to find.</param>
public T FindFirstChildOfClass<T>(bool recursive = false) where T : Instance
{
    var query = _children
        .Where(child => child is T)
        .Cast<T>();

    T result = null;
    
    if (query.Any())
    {
        result = query.First();
    }
    else if (recursive)
    {
        foreach (Instance child in _children)
        {
            T found = child.FindFirstChildOfClass<T>(true);

            if (found != null)
            {
                result = found;
                break;
            }
        }
    }

    return result;
}

    /// <summary>
/// Disposes of this instance and its descendants, and locks its parent.
/// All property bindings, tags, and attributes are cleared.
    /// </summary>
public void Destroy()
{
    Destroyed = true;
    props.Clear();
    
    Parent = null;
    _parent_locked = true;

    Tags.Clear();
    Attributes.Clear();

    while (_children.Any())
    {
        var child = _children.First();
        child.Destroy();
    }
}

    /// <summary>
/// Creates a deep copy of this instance and its descendants.
/// Any instances that have Archivable set to false are not included.
/// This can include the instance itself, in which case this will return null.
    /// </summary>
public Instance Clone()
{
    var mitosis = new Dictionary<Instance, Instance>();
    var refProps = new List<Property>();

    var insts = GetDescendants().ToList();
    insts.Insert(0, this);

    foreach (var oldInst in insts)
    {
        if (!oldInst.Archivable)
            continue;

        var type = oldInst.GetType();
        var newInst = Activator.CreateInstance(type) as Instance;

        foreach (var pair in oldInst.Properties)
        {
            // Create memberwise copy of the property.
            var oldProp = pair.Value;

            var newProp = new Property()
            {
                Instance = newInst,

                Name = oldProp.Name,
                Type = oldProp.Type,

                Value = oldProp.Value,
                XmlToken = oldProp.XmlToken,
            };

            if (newProp.Type == PropertyType.Ref)
                refProps.Add(newProp);

            newInst.AddProperty(ref newProp);
        }

        var oldParent = oldInst.Parent;
        mitosis[oldInst] = newInst;

        if (oldParent == null)
            continue;

        if (!mitosis.TryGetValue(oldParent, out var newParent))
            continue;

        newInst.Parent = newParent;
    }

    // Patch referents where applicable.
    foreach (var prop in refProps)
    {
        if (!(prop.Value is Instance source))
            continue;

        if (!mitosis.TryGetValue(source, out var copy))
            continue;

        prop.Value = copy;
    }

    // Grab the copy of ourselves that we created.
    mitosis.TryGetValue(this, out Instance clone);

    return clone;
}

    /// <summary>
/// Returns the first child of this Instance which derives from the provided type <typeparamref name="T"/>.
/// If the instance is not found, this returns null.
    /// </summary>
/// <param name="recursive">Whether this should search descendants as well.</param>
public T FindFirstChildWhichIsA<T>(bool recursive = false) where T : Instance
{
    var query = _children
        .Where(child => child is T)
        .Cast<T>();

    if (query.Any())
        return query.First();
    
    if (recursive)
    {
        foreach (Instance child in _children)
        {
            T found = child.FindFirstChildWhichIsA<T>(true);

            if (found == null)
                continue;

            return found;
        }
    }

    return null;
}
"""

    def GetFullName(self, separator: str = "\\") -> str:
        """
        :return: a string describing the index traversal of this Instance, starting from its root ancestor.
        """
        full_name = self.Name
        at = self.Parent

        while at is not None:
            full_name = at.Name + separator + full_name
            at = at.Parent

        return full_name


"""
    /// <summary>
/// Returns a Property object whose name is the provided string name.
    /// </summary>
public Property GetProperty(string name)
{
    Property result = null;

    if (props.ContainsKey(name))
        result = props[name];

    return result;
}

    /// <summary>
/// Adds a property by reference to this Instance's property list.
    /// </summary>
/// <param name="prop">A reference to the property that will be added.</param>
internal void AddProperty(ref Property prop)
{
    string name = prop.Name;
    RemoveProperty(name);

    prop.Instance = this;
    props.Add(name, prop);
}

    /// <summary>
/// Removes a property with the provided name if a property with the provided name exists.
    /// </summary>
/// <param name="name">The name of the property to be removed.</param>
/// <returns>True if a property with the provided name was removed.</returns> 
internal bool RemoveProperty(string name)
{
    if (props.ContainsKey(name))
    {
        Property prop = Properties[name];
        prop.Instance = null;
    }

    return props.Remove(name);
}

    /// <summary>
/// Ensures that all serializable properties of this Instance have
/// a registered Property object with the correct PropertyType.
    /// </summary>
internal IReadOnlyDictionary<string, Property> RefreshProperties()
{
    Type instType = GetType();
    FieldInfo[] fields = instType.GetFields(Property.BindingFlags);

    foreach (FieldInfo field in fields)
    {
        string fieldName = field.Name;
        Type fieldType = field.FieldType;

        // A few specific edge case hacks. I wish these didn't need to exist :(
        if (fieldName == "Archivable" || fieldName.EndsWith("k__BackingField"))
            continue;
        else if (fieldName == "Bevel_Roundness")
            fieldName = "Bevel Roundness";

        PropertyType propType = PropertyType.Unknown;

        if (fieldType.IsEnum)
            propType = PropertyType.Enum;
        else if (Property.Types.ContainsKey(fieldType))
            propType = Property.Types[fieldType];
        else if (typeof(Instance).IsAssignableFrom(fieldType))
            propType = PropertyType.Ref;

        if (propType != PropertyType.Unknown)
        {
            if (fieldName.EndsWith("_"))
                fieldName = instType.Name;

            string xmlToken = fieldType.Name;

            if (fieldType.IsEnum)
                xmlToken = "token";
            else if (propType == PropertyType.Ref)
                xmlToken = "Ref";

            switch (xmlToken)
            {
                case "String":
                case "Double":
                case "Int64":
                {
                    xmlToken = xmlToken.ToLowerInvariant();
                    break;
                }
                case "Boolean":
                {
                    xmlToken = "bool";
                    break;
                }
                case "Single":
                {
                    xmlToken = "float";
                    break;
                }
                case "Int32":
                {
                    xmlToken = "int";
                    break;
                }
                case "Rect":
                {
                    xmlToken = "Rect2D";
                    break;
                }
                case "CFrame":
                {
                    xmlToken = "CoordinateFrame";
                    break;
                }
                case "FontFace":
                {
                    xmlToken = "Font";
                    break;
                }
                case "Optional`1":
                {
                    // TODO: If more optional types are added,
                    //       this needs disambiguation.

                    xmlToken = "OptionalCoordinateFrame";
                    break;
                }
            }

            if (!props.ContainsKey(fieldName))
            {
                var newProp = new Property()
                {
                    Value = field.GetValue(this),
                    XmlToken = xmlToken,
                    Name = fieldName,
                    Type = propType,
                    Instance = this
                };

                AddProperty(ref newProp);
            }
            else
            {
                Property prop = props[fieldName];
                prop.Value = field.GetValue(this);
                prop.XmlToken = xmlToken;
                prop.Type = propType;
            }
        }
    }

    Property tags = GetProperty("Tags");
    Property attributes = GetProperty("AttributesSerialize");
    
    if (tags == null)
    {
        tags = new Property("Tags", PropertyType.String);
        AddProperty(ref tags);
    }

    if (attributes == null)
    {
        attributes = new Property("AttributesSerialize", PropertyType.String);
        AddProperty(ref attributes);
    }

    return Properties;
}
}
}
"""
