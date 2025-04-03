#!/usr/bin/env python3
"""
Simple Documentation Generator

Extracts documentation from Python code and generates HTML or Markdown documentation.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import sys
import re
import ast
import inspect
import importlib
import argparse
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.utils.logging_framework import get_logger

logger = get_logger("doc_generator")


class DocGenerator:
    """Simple documentation generator."""
    
    def __init__(self, 
                 output_dir: str = "/Users/yessine/Oblivion/docs",
                 format: str = "markdown"):
        """
        Initialize documentation generator.
        
        Args:
            output_dir: Directory to store generated documentation
            format: Output format (markdown or html)
        """
        self.output_dir = output_dir
        self.format = format.lower()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized documentation generator with output to {self.output_dir}")
    
    def generate_docs_for_file(self, file_path: str) -> Dict[str, Any]:
        """
        Generate documentation for a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dict[str, Any]: Documentation data
        """
        if not os.path.exists(file_path) or not file_path.endswith(".py"):
            logger.warning(f"Invalid Python file: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            # Parse the code
            tree = ast.parse(code)
            
            # Extract module docstring
            module_doc = ast.get_docstring(tree)
            
            # Extract classes and functions
            classes = []
            functions = []
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    methods = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_doc = ast.get_docstring(item)
                            methods.append({
                                "name": item.name,
                                "docstring": method_doc,
                                "args": [arg.arg for arg in item.args.args if arg.arg != "self"],
                                "line": item.lineno
                            })
                    
                    classes.append({
                        "name": node.name,
                        "docstring": class_doc,
                        "methods": methods,
                        "line": node.lineno
                    })
                
                elif isinstance(node, ast.FunctionDef):
                    func_doc = ast.get_docstring(node)
                    functions.append({
                        "name": node.name,
                        "docstring": func_doc,
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno
                    })
            
            # Create documentation data
            doc_data = {
                "file_path": file_path,
                "module_name": os.path.basename(file_path).replace(".py", ""),
                "module_docstring": module_doc,
                "classes": classes,
                "functions": functions
            }
            
            return doc_data
            
        except Exception as e:
            logger.error(f"Error generating documentation for {file_path}: {str(e)}")
            return {}
    
    def generate_docs_for_module(self, module_path: str) -> List[Dict[str, Any]]:
        """
        Generate documentation for a Python module.
        
        Args:
            module_path: Path to Python module
            
        Returns:
            List[Dict[str, Any]]: Documentation data for all files in the module
        """
        if not os.path.exists(module_path):
            logger.warning(f"Module path not found: {module_path}")
            return []
        
        doc_data = []
        
        if os.path.isfile(module_path) and module_path.endswith(".py"):
            # Single file
            data = self.generate_docs_for_file(module_path)
            if data:
                doc_data.append(data)
        
        elif os.path.isdir(module_path):
            # Directory
            for root, _, files in os.walk(module_path):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        file_path = os.path.join(root, file)
                        data = self.generate_docs_for_file(file_path)
                        if data:
                            doc_data.append(data)
        
        return doc_data
    
    def generate_markdown(self, doc_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate Markdown documentation.
        
        Args:
            doc_data: Documentation data
            
        Returns:
            Dict[str, str]: Dictionary of file paths and their Markdown content
        """
        markdown_files = {}
        
        for data in doc_data:
            file_path = data["file_path"]
            module_name = data["module_name"]
            
            # Create Markdown content
            content = f"# {module_name}\n\n"
            
            if data["module_docstring"]:
                content += f"{data['module_docstring']}\n\n"
            
            # Add classes
            if data["classes"]:
                content += "## Classes\n\n"
                
                for cls in data["classes"]:
                    content += f"### {cls['name']}\n\n"
                    
                    if cls["docstring"]:
                        content += f"{cls['docstring']}\n\n"
                    
                    # Add methods
                    if cls["methods"]:
                        for method in cls["methods"]:
                            content += f"#### {method['name']}({', '.join(method['args'])})\n\n"
                            
                            if method["docstring"]:
                                content += f"{method['docstring']}\n\n"
            
            # Add functions
            if data["functions"]:
                content += "## Functions\n\n"
                
                for func in data["functions"]:
                    content += f"### {func['name']}({', '.join(func['args'])})\n\n"
                    
                    if func["docstring"]:
                        content += f"{func['docstring']}\n\n"
            
            # Add to dictionary
            rel_path = os.path.relpath(file_path, "/Users/yessine/Oblivion")
            output_path = os.path.join(self.output_dir, f"{rel_path.replace('/', '_').replace('.py', '.md')}")
            markdown_files[output_path] = content
        
        return markdown_files
    
    def generate_html(self, doc_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate HTML documentation.
        
        Args:
            doc_data: Documentation data
            
        Returns:
            Dict[str, str]: Dictionary of file paths and their HTML content
        """
        html_files = {}
        
        # Generate index.html
        index_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Oblivion Documentation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                h1 { color: #333; }
                h2 { color: #444; margin-top: 30px; }
                h3 { color: #555; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
                .sidebar { width: 250px; position: fixed; top: 0; left: 0; height: 100%; overflow: auto; background-color: #f8f8f8; padding: 20px; }
                .content { margin-left: 290px; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="sidebar">
                <h2>Modules</h2>
                <ul>
        """
        
        # Add modules to index
        for data in doc_data:
            module_name = data["module_name"]
            rel_path = os.path.relpath(data["file_path"], "/Users/yessine/Oblivion")
            html_filename = f"{rel_path.replace('/', '_').replace('.py', '.html')}"
            index_content += f'        <li><a href="{html_filename}">{module_name}</a></li>\n'
        
        index_content += """
                </ul>
            </div>
            <div class="content">
                <h1>Oblivion Documentation</h1>
                <p>Welcome to the documentation for the Oblivion neuromorphic system.</p>
                <p>Select a module from the sidebar to view its documentation.</p>
            </div>
        </body>
        </html>
        """
        
        html_files[os.path.join(self.output_dir, "index.html")] = index_content
        
        # Generate HTML for each module
        for data in doc_data:
            file_path = data["file_path"]
            module_name = data["module_name"]
            
            content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{module_name} - Oblivion Documentation</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #444; margin-top: 30px; }}
                    h3 {{ color: #555; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    .sidebar {{ width: 250px; position: fixed; top: 0; left: 0; height: 100%; overflow: auto; background-color: #f8f8f8; padding: 20px; }}
                    .content {{ margin-left: 290px; }}
                    a {{ color: #0066cc; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                </style>
            </head>
            <body>
                <div class="sidebar">
                    <h2>Modules</h2>
                    <ul>
            """
            
            # Add modules to sidebar
            for d in doc_data:
                m_name = d["module_name"]
                r_path = os.path.relpath(d["file_path"], "/Users/yessine/Oblivion")
                h_filename = f"{r_path.replace('/', '_').replace('.py', '.html')}"
                
                if m_name == module_name:
                    content += f'        <li><strong>{m_name}</strong></li>\n'
                else:
                    content += f'        <li><a href="{h_filename}">{m_name}</a></li>\n'
            
            content += f"""
                    </ul>
                </div>
                <div class="content">
                    <h1>{module_name}</h1>
            """
            
            if data["module_docstring"]:
                content += f"    <p>{data['module_docstring']}</p>\n"
            
            # Add classes
            if data["classes"]:
                content += "    <h2>Classes</h2>\n"
                
                for cls in data["classes"]:
                    content += f"    <h3>{cls['name']}</h3>\n"
                    
                    if cls["docstring"]:
                        content += f"    <p>{cls['docstring']}</p>\n"
                    
                    # Add methods
                    if cls["methods"]:
                        for method in cls["methods"]:
                            content += f"    <h4>{method['name']}({', '.join(method['args'])})</h4>\n"
                            
                            if method["docstring"]:
                                content += f"    <p>{method['docstring']}</p>\n"
            
            # Add functions
            if data["functions"]:
                content += "    <h2>Functions</h2>\n"
                
                for func in data["functions"]:
                    content += f"    <h3>{func['name']}({', '.join(func['args'])})</h3>\n"
                    
                    if func["docstring"]:
                        content += f"    <p>{func['docstring']}</p>\n"
            
            content += """
                </div>
            </body>
            </html>
            """
            
            # Add to dictionary
            rel_path = os.path.relpath(file_path, "/Users/yessine/Oblivion")
            output_path = os.path.join(self.output_dir, f"{rel_path.replace('/', '_').replace('.py', '.html')}")
            html_files[output_path] = content
        
        return html_files
    
    def write_docs(self, doc_files: Dict[str, str]) -> None:
        """
        Write documentation files.
        
        Args:
            doc_files: Dictionary of file paths and their content
        """
        for file_path, content in doc_files.items():
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file
            with open(file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Wrote documentation to {file_path}")
    
    def generate(self, module_paths: List[str]) -> bool:
        """
        Generate documentation for modules.
        
        Args:
            module_paths: List of module paths
            
        Returns:
            bool: True if documentation was generated successfully
        """
        try:
            # Generate documentation data
            all_doc_data = []
            
            for module_path in module_paths:
                doc_data = self.generate_docs_for_module(module_path)
                all_doc_data.extend(doc_data)
            
            if not all_doc_data:
                logger.warning("No documentation data generated")
                return False
            
            # Generate documentation files
            if self.format == "markdown":
                doc_files = self.generate_markdown(all_doc_data)
            else:
                doc_files = self.generate_html(all_doc_data)
            
            # Write documentation files
            self.write_docs(doc_files)
            
            logger.info(f"Generated documentation for {len(all_doc_data)} modules")
            return True
            
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            return False


def main():
    """Main entry point for documentation generator."""
    parser = argparse.ArgumentParser(description="Simple Documentation Generator")
    parser.add_argument("--output-dir", default="/Users/yessine/Oblivion/docs", help="Output directory")
    parser.add_argument("--format", choices=["markdown", "html"], default="markdown", help="Output format")
    parser.add_argument("modules", nargs="*", default=["/Users/yessine/Oblivion/src"], help="Modules to document")
    
    args = parser.parse_args()
    
    # Create documentation generator
    generator = DocGenerator(
        output_dir=args.output_dir,
        format=args.format
    )
    
    # Generate documentation
    success = generator.generate(args.modules)
    
    if success:
        print(f"Documentation generated in {args.output_dir}")
    else:
        print("Failed to generate documentation")
        sys.exit(1)


if __name__ == "__main__":
    main()