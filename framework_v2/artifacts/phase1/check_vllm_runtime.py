import argparse
import sys
from importlib.metadata import PackageNotFoundError, distributions, requires, version

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def _installed_versions() -> dict[str, str]:
    installed = {}
    for dist in distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            continue
        if name:
            installed[canonicalize_name(name)] = dist.version
    return installed


def _check_dependency(requirement_text: str, installed: dict[str, str]) -> str | None:
    req = Requirement(requirement_text)
    if req.marker and not req.marker.evaluate():
        return None

    name = canonicalize_name(req.name)
    actual = installed.get(name)
    if actual is None:
        return f"{req.name} is not installed (required: {req.specifier or 'any'})"

    if req.specifier and not req.specifier.contains(Version(actual), prereleases=True):
        return f"{req.name}=={actual} does not satisfy {req.specifier}"

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that the active Python environment satisfies vLLM runtime requirements."
    )
    parser.add_argument("--package", default="vllm", help="Distribution name to validate.")
    args = parser.parse_args()

    try:
        pkg_version = version(args.package)
    except PackageNotFoundError:
        print(f"{args.package} is not installed in the active environment.", file=sys.stderr)
        return 2

    installed = _installed_versions()
    problems = []
    for dep in requires(args.package) or []:
        problem = _check_dependency(dep, installed)
        if problem:
            problems.append(problem)

    print(f"{args.package}=={pkg_version}")
    if not problems:
        print("Dependency check: OK")
        return 0

    print("Dependency check: FAILED")
    for item in problems:
        print(f"- {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
