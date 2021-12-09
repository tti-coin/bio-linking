import argparse as _A
import dataclasses as _D
from typing import TypeVar, Type, Union, Generic, Optional

def get_origin(type_):
    return type_.__origin__
class TypingType:
    def __init__(self, type_):
        self.type = type_
    @property
    def is_union(self):
        return (hasattr(self.type, "__origin__")) and (get_origin(self.type) is Union)
    @property
    def is_optional(self):
        return self.is_union and (len(self.type.__args__) == 2) and (self.type.__args__[1] is type(None))
    @property
    def optional_arg_type(self):
        return self.type.__args__[0]

Dataclass = TypeVar("Dataclass")
class DataclassArgumentParser(_A.ArgumentParser, Generic[Dataclass]):
    """
    variable name: treated as "dest"
    metadata:
    - args: treated as aliases.
    - default: default value. priority=> field.default > field.default_factory > field.metadata["default"]

    - choices: choices
    - required: required
    - help: help

    - type, dest, action: prohibited.
    """
    def __init__(self, dataclass:Type[Dataclass], default_value:Optional[Dataclass]=None, *args, **kwargs):
        super(DataclassArgumentParser, self).__init__(*args, **kwargs)
        self._dataclass = dataclass
        self._default_value = default_value

        self._add_argument_from_dataclass()
        if self._default_value is not None:
            self.set_defaults(**_D.asdict(self._default_value))

    def _add_argument_from_dataclass(self) -> None:
        for field in _D.fields(self._dataclass):
            argparams = list()

            assert "dest" not in field.metadata, field
            assert "type" not in field.metadata, field
            assert "action" not in field.metadata, field
            if "args" in field.metadata:
                assert type(field.metadata["args"]) in {list, tuple}, field
                assert all(a[0] == "-" for a in field.metadata["args"]), f"must be a non-positional argument: {field}"

            doesnt_have_default = (
                (field.default == _D.MISSING)
                and (field.default_factory == _D.MISSING)
                and ("default" not in field.metadata)
            )
            if not doesnt_have_default:
                default_value = (
                    field.default if field.default != _D.MISSING
                    else field.default_factory() if field.default_factory != _D.MISSING
                    else field.metadata["default"]
                )

            type_ = field.type
            ttype = TypingType(type_)
            if ttype.is_optional:
                assert (not doesnt_have_default) and (default_value is None), field
                type_ = ttype.optional_arg_type
                assert type_ is not bool

            if type_ is bool:
                assert not doesnt_have_default, field
                assert type(default_value) is bool
                assert "args" not in field.metadata, field
                assert "choices" not in field.metadata, field
                assert "required" not in field.metadata, field

                true_argkwparams = dict()
                false_argkwparams = dict()

                if "help" in field.metadata:
                    true_argkwparams["help"] = field.metadata["help"]

                default_side_argkwparams = true_argkwparams if default_value is True else false_argkwparams
                default_side_argkwparams["help"] = " ".join(["(default)"] + default_side_argkwparams.get("help", []))

                self.add_argument("--"+field.name, dest=field.name, action="store_true", **true_argkwparams)
                self.add_argument("--no_"+field.name, dest=field.name, action="store_false", **false_argkwparams)
                self.set_defaults(**{field.name:default_value})

            else:
                argkwparams = dict()

                if "help" in field.metadata:
                    argkwparams["help"] = field.metadata["help"]
                else:
                    argkwparams["help"] = f"{type_}"
                if ttype.is_optional:
                    argkwparams["help"] = " ".join([argkwparams["help"], "(optional)"])
                elif not doesnt_have_default:
                    argkwparams["help"] = " ".join([argkwparams["help"], f"(default={default_value})"])

                argkwparams["type"] = type_

                is_positional = ("args" not in field.metadata) and doesnt_have_default

                if is_positional:
                    raise NotImplementedError()

                else:
                    argkwparams["dest"] = field.name
                    if "args" in field.metadata:
                        argparams = field.metadata["args"]
                    else:
                        argparams = ["--"+field.name]
                    for key in ["choices", "required"]:
                        if key in field.metadata:
                            argkwparams[key] = field.metadata[key]
                    if not doesnt_have_default:
                        argkwparams["default"] = default_value

                self.add_argument(*argparams, **argkwparams)

    def parse_args(self, *args, **kwargs) -> Dataclass:
        namespace_args = super(DataclassArgumentParser, self).parse_args(*args, **kwargs)
        return self._dataclass(**vars(namespace_args))

