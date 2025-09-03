#!/usr/bin/env python
"""
Dependency Analyzer for Thea Code System
Analyzes imports, dependencies, and connectivity to make informed wiring decisions

Uses Python's ast module to parse files and extract import information
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class ImportInfo:
    """Information about an import statement"""
    module: str
    names: List[str]  # For 'from x import a, b'
    alias: Optional[str]  # For 'import x as y'
    is_relative: bool  # True for relative imports like 'from .module import x'
    line_number: int


@dataclass
class FileAnalysis:
    """Analysis results for a single Python file"""
    file_path: str
    imports: List[ImportInfo]
    dependencies: Set[str]  # All modules this file depends on
    exports: Set[str]  # All names this file exports (classes, functions, etc.)
    errors: List[str]  # Parsing errors


class DependencyAnalyzer:
    """
    Analyzes Python code dependencies using AST parsing
    
    This helps us understand:
    - What imports what
    - Circular dependencies  
    - Missing dependencies
    - Unused imports
    - Export/import relationships
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.analyses: Dict[str, FileAnalysis] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single Python file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            imports = []
            dependencies = set()
            exports = set()
            
            # Walk the AST to find imports and exports
            for node in ast.walk(tree):
                # Import statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_info = ImportInfo(
                            module=alias.name,
                            names=[],
                            alias=alias.asname,
                            is_relative=False,
                            line_number=node.lineno
                        )
                        imports.append(import_info)
                        dependencies.add(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""  # Handle 'from . import x'
                    names = [alias.name for alias in node.names]
                    
                    import_info = ImportInfo(
                        module=module,
                        names=names,
                        alias=None,
                        is_relative=node.level > 0,
                        line_number=node.lineno
                    )
                    imports.append(import_info)
                    
                    if module:
                        dependencies.add(module)
                
                # Exports (top-level classes, functions, variables)
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    exports.add(node.name)
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            exports.add(target.id)
            
            return FileAnalysis(
                file_path=str(file_path.relative_to(self.root_dir)),
                imports=imports,
                dependencies=dependencies,
                exports=exports,
                errors=[]
            )
            
        except Exception as e:
            return FileAnalysis(
                file_path=str(file_path.relative_to(self.root_dir)),
                imports=[],
                dependencies=set(),
                exports=set(),
                errors=[str(e)]
            )
    
    def analyze_directory(self, target_dir: Optional[str] = None) -> None:
        """Analyze all Python files in directory"""
        
        if target_dir:
            scan_dir = self.root_dir / target_dir
        else:
            scan_dir = self.root_dir
            
        print(f"🔍 Analyzing Python files in {scan_dir}")
        
        # Find all Python files
        python_files = list(scan_dir.rglob("*.py"))
        print(f"📁 Found {len(python_files)} Python files")
        
        # Analyze each file
        for file_path in python_files:
            # Skip __pycache__ and other generated directories
            if "__pycache__" in str(file_path) or ".git" in str(file_path):
                continue
                
            analysis = self.analyze_file(file_path)
            self.analyses[analysis.file_path] = analysis
            
            # Build dependency graph
            for dep in analysis.dependencies:
                self.dependency_graph[analysis.file_path].add(dep)
                self.reverse_deps[dep].add(analysis.file_path)
        
        print(f"✅ Analyzed {len(self.analyses)} files")
    
    def find_internal_dependencies(self) -> Dict[str, Set[str]]:
        """Find dependencies between our own modules"""
        
        internal_deps = defaultdict(set)
        
        for file_path, analysis in self.analyses.items():
            for import_info in analysis.imports:
                # Check if this is an internal import
                if import_info.is_relative:
                    # Relative import like 'from .models import X'
                    internal_deps[file_path].add(f"relative:{import_info.module}")
                
                elif any(part in import_info.module for part in ["thea_code_system", "actors", "models", "utils"]):
                    # Absolute import of our own modules
                    internal_deps[file_path].add(import_info.module)
        
        return internal_deps
    
    def find_external_dependencies(self) -> Dict[str, int]:
        """Find external library dependencies"""
        
        external_deps = defaultdict(int)
        
        for analysis in self.analyses.values():
            for import_info in analysis.imports:
                # Skip relative imports and our own modules
                if (import_info.is_relative or 
                    any(part in import_info.module for part in ["thea_code_system", "actors", "models", "utils"])):
                    continue
                
                # Count external dependencies
                root_module = import_info.module.split('.')[0]
                external_deps[root_module] += 1
        
        return external_deps
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies using DFS"""
        
        def dfs(node, path, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor not in self.analyses:  # Skip external deps
                    continue
                    
                if neighbor not in visited:
                    result = dfs(neighbor, path, visited, rec_stack)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            path.pop()
            rec_stack.remove(node)
            return None
        
        visited = set()
        cycles = []
        
        for file_path in self.analyses:
            if file_path not in visited:
                cycle = dfs(file_path, [], visited, set())
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def find_missing_dependencies(self) -> Dict[str, List[str]]:
        """Find imports that might be missing or broken"""
        
        missing = defaultdict(list)
        
        for file_path, analysis in self.analyses.items():
            for import_info in analysis.imports:
                # Check if imported module exists in our codebase
                if import_info.is_relative:
                    # For relative imports, resolve the actual module path
                    # This is simplified - real resolution would be more complex
                    continue
                
                # Check if it's a standard library or external dependency
                try:
                    __import__(import_info.module)
                except ImportError:
                    missing[file_path].append(import_info.module)
        
        return missing
    
    def generate_report(self) -> str:
        """Generate comprehensive dependency report"""
        
        # Calculate statistics
        total_files = len(self.analyses)
        total_imports = sum(len(a.imports) for a in self.analyses.values())
        files_with_errors = sum(1 for a in self.analyses.values() if a.errors)
        
        # Get dependency information
        internal_deps = self.find_internal_dependencies()
        external_deps = self.find_external_dependencies()
        circular_deps = self.find_circular_dependencies()
        missing_deps = self.find_missing_dependencies()
        
        report = f"""
🔍 THEA CODE SYSTEM - DEPENDENCY ANALYSIS
{'='*50}

📊 SUMMARY
Files analyzed: {total_files}
Total imports: {total_imports}
Files with errors: {files_with_errors}
Internal dependencies: {sum(len(deps) for deps in internal_deps.values())}
External dependencies: {len(external_deps)}

🏗️ EXTERNAL DEPENDENCIES
Top external libraries:
"""
        
        # Sort external deps by usage count
        sorted_external = sorted(external_deps.items(), key=lambda x: x[1], reverse=True)
        for lib, count in sorted_external[:15]:  # Top 15
            report += f"  {lib:20s} ({count:2d} imports)\n"
        
        report += f"""

🔗 INTERNAL DEPENDENCY STRUCTURE
"""
        
        # Show internal dependency structure
        if internal_deps:
            report += "Internal imports:\n"
            for file_path, deps in sorted(internal_deps.items()):
                if deps:
                    report += f"  {file_path}:\n"
                    for dep in sorted(deps):
                        report += f"    → {dep}\n"
        else:
            report += "  No internal dependencies found\n"
        
        # Circular dependencies
        report += f"""

🔄 CIRCULAR DEPENDENCIES
"""
        if circular_deps:
            for i, cycle in enumerate(circular_deps, 1):
                report += f"  Cycle {i}: {' → '.join(cycle)}\n"
        else:
            report += "  ✅ No circular dependencies found\n"
        
        # Missing dependencies
        report += f"""

❌ POTENTIAL MISSING DEPENDENCIES
"""
        if missing_deps:
            for file_path, missing in missing_deps.items():
                if missing:
                    report += f"  {file_path}:\n"
                    for dep in missing:
                        report += f"    ❌ {dep}\n"
        else:
            report += "  ✅ No missing dependencies detected\n"
        
        # File-by-file breakdown
        report += f"""

📁 FILE-BY-FILE ANALYSIS
"""
        
        for file_path, analysis in sorted(self.analyses.items()):
            report += f"\n📄 {file_path}\n"
            report += f"   Imports: {len(analysis.imports)}\n"
            report += f"   Exports: {len(analysis.exports)} ({', '.join(sorted(list(analysis.exports)[:5]))}{'...' if len(analysis.exports) > 5 else ''})\n"
            
            if analysis.errors:
                report += f"   ❌ Errors: {', '.join(analysis.errors)}\n"
            
            # Show key imports
            key_imports = [imp for imp in analysis.imports if not imp.module.startswith('__')][:3]
            if key_imports:
                report += f"   Key imports: {', '.join(imp.module for imp in key_imports)}\n"
        
        return report
    
    def export_json(self, output_file: str) -> None:
        """Export analysis results to JSON"""
        
        data = {
            'summary': {
                'total_files': len(self.analyses),
                'total_imports': sum(len(a.imports) for a in self.analyses.values()),
                'files_with_errors': sum(1 for a in self.analyses.values() if a.errors)
            },
            'external_dependencies': dict(self.find_external_dependencies()),
            'internal_dependencies': {k: list(v) for k, v in self.find_internal_dependencies().items()},
            'circular_dependencies': self.find_circular_dependencies(),
            'missing_dependencies': {k: list(v) for k, v in self.find_missing_dependencies().items()},
            'files': {}
        }
        
        # Add file details
        for file_path, analysis in self.analyses.items():
            data['files'][file_path] = {
                'imports': [
                    {
                        'module': imp.module,
                        'names': imp.names,
                        'alias': imp.alias,
                        'is_relative': imp.is_relative,
                        'line_number': imp.line_number
                    }
                    for imp in analysis.imports
                ],
                'exports': list(analysis.exports),
                'errors': analysis.errors
            }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Analysis exported to {output_file}")


def main():
    """Main entry point for dependency analysis"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Python dependencies")
    parser.add_argument("--directory", "-d", default="thea_code_system", help="Directory to analyze")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DependencyAnalyzer(".")
    
    # Analyze directory
    analyzer.analyze_directory(args.directory)
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Export JSON if requested
    if args.output:
        analyzer.export_json(args.output)
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("="*50)
    
    external_deps = analyzer.find_external_dependencies()
    internal_deps = analyzer.find_internal_dependencies()
    circular_deps = analyzer.find_circular_dependencies()
    
    if len(external_deps) > 20:
        print("⚠️  High number of external dependencies - consider consolidation")
    
    if circular_deps:
        print("❌ Fix circular dependencies before production")
    else:
        print("✅ No circular dependencies - good architecture")
    
    if len(internal_deps) == 0:
        print("⚠️  No internal dependencies detected - check import paths")
    else:
        print(f"✅ {len(internal_deps)} files have internal dependencies - good modular structure")
    
    print("\n🚀 Ready for informed wiring and connectivity decisions!")


if __name__ == "__main__":
    main()