"""
Tool for discovering available document folders with hierarchical numbering.
"""

from typing import Any, Dict, List

from langchain_core.tools import tool

from ..logging_config import get_logger
from ._common import get_chunks_dir

logger = get_logger(__name__)


@tool
def discover_documents(product_number: str = None) -> str:
    """
    Discovers available document folders with hierarchical numbering (1, 1.1, 1.1.1).
    
    Args:
        product_number: Optional product number (e.g., "1", "2.1") to explore subdirectories.
                       If None, returns top-level product list.
    
    Returns:
        Numbered list of products/documents with mapping between numbers and paths.
        Use the returned numbers in subsequent calls to explore deeper levels.
    """
    logger.info(
        "discover_documents called",
        extra={"extra_fields": {"product_number": product_number}},
    )
    _chunks_dir = get_chunks_dir()
    if not _chunks_dir.exists():
        logger.info(
            "discover_documents done",
            extra={
                "extra_fields": {
                    "product_number": product_number,
                    "outcome": "error",
                    "summary": "chunks_dir_not_found",
                }
            },
        )
        return f"Error: Chunks directory \"{_chunks_dir}\" does not exist."

    # CASE 1: Top-level Product List
    if product_number is None:
        products = sorted([d.name for d in _chunks_dir.iterdir() if d.is_dir()])
        output = ["Available Products (use number to explore):"]
        output.append("")
        
        for idx, prd in enumerate(products, 1):
            output.append(f"{idx}. {prd}")
        
        output.append("")
        output.append("To explore a product's documents, call discover_documents(product_number=\"1\") or the appropriate number.")
        result = "\n".join(output)
        logger.info(
            "discover_documents done",
            extra={
                "extra_fields": {
                    "product_number": product_number,
                    "outcome": "top_level",
                    "summary": f"{len(products)} products",
                }
            },
        )
        return result

    # Parse the product number to get the path
    # Number format: "1" -> first product, "1.1" -> first subdoc of first product
    parts = product_number.split(".")
    
    # Get all products for mapping
    products = sorted([d.name for d in _chunks_dir.iterdir() if d.is_dir()])
    
    if not parts or not parts[0].isdigit():
        logger.info(
            "discover_documents done",
            extra={
                "extra_fields": {
                    "product_number": product_number,
                    "outcome": "error",
                    "summary": "invalid_format",
                }
            },
        )
        return f"Error: Invalid product_number format. Use numbers like '1', '2', '1.1', etc."

    # Navigate to the requested level
    try:
        product_idx = int(parts[0]) - 1
        if product_idx < 0 or product_idx >= len(products):
            logger.info(
                "discover_documents done",
                extra={
                    "extra_fields": {
                        "product_number": product_number,
                        "outcome": "error",
                        "summary": "product_out_of_range",
                    }
                },
            )
            return f"Error: Product number {parts[0]} out of range (1-{len(products)})."

        current_path = _chunks_dir / products[product_idx]
        
        # Navigate deeper if there are more parts
        for part_idx, part in enumerate(parts[1:], 1):
            if not part.isdigit():
                return f"Error: Invalid number format in '{product_number}'."
            
            # Get subdirectories at current level
            subdirs = sorted([d for d in current_path.iterdir() if d.is_dir() and (d / "chunk").exists()])
            
            sub_idx = int(part) - 1
            if sub_idx < 0 or sub_idx >= len(subdirs):
                logger.info(
                    "discover_documents done",
                    extra={
                        "extra_fields": {
                            "product_number": product_number,
                            "outcome": "error",
                            "summary": "subdir_out_of_range",
                        }
                    },
                )
                return f"Error: Subdirectory number {part} out of range at level {'.'.join(parts[:part_idx+1])}."

            current_path = subdirs[sub_idx]

    except (ValueError, IndexError) as e:
        logger.info(
            "discover_documents done",
            extra={
                "extra_fields": {
                    "product_number": product_number,
                    "outcome": "error",
                    "summary": str(e),
                }
            },
        )
        return f"Error: Invalid product_number format: {e}"
    
    # Build the tree structure from current_path
    def get_tree(): return {}
    tree = get_tree()
    
    for chunk_path in current_path.rglob("chunk"):
        rel_path = chunk_path.relative_to(current_path)
        parts_list = [p for p in rel_path.parts if p != "chunk"]
        
        current_node = tree
        for part in parts_list:
            if part not in current_node:
                current_node[part] = get_tree()
            current_node = current_node[part]
    
    # Format with hierarchical numbering
    def format_tree_numbered(node: Dict[str, Any], prefix: str = "") -> List[str]:
        lines = []
        sorted_keys = sorted(node.keys())
        
        for idx, key in enumerate(sorted_keys, 1):
            if prefix:
                number = f"{prefix}.{idx}"
            else:
                number = str(idx)
            
            # Get the full path for this item
            lines.append(f"{number}. {key}")
            
            # Recurse into children
            child_lines = format_tree_numbered(node[key], number)
            lines.extend(child_lines)
        
        return lines
    
    # Assemble output
    relative_path = current_path.relative_to(_chunks_dir)
    output = [f"Documents in '{relative_path}' (Number: {product_number}):"]
    output.append("")
    
    formatted_tree = format_tree_numbered(tree)
    
    if not formatted_tree:
        output.append("No subdocuments found at this level.")
        output.append(f"This is a leaf document.")
        output.append("")
        output.append(f"SECTION NUMBER: {product_number}")
        output.append(f"DOCUMENT PATH: {relative_path}")
        outcome = "leaf"
        summary = str(relative_path)
    else:
        output.extend(formatted_tree)
        output.append("")
        output.append(f"To explore deeper, call discover_documents(product_number=\"{product_number}.X\") where X is the subdocument number.")
        output.append("")
        output.append("=== SECTION NUMBER → DOCUMENT PATH MAPPING ===")
        # Build mapping for all leaf documents at this level
        for idx, key in enumerate(sorted(tree.keys()), 1):
            if product_number:
                section_num = f"{product_number}.{idx}"
            else:
                section_num = str(idx)
            doc_path = relative_path / key
            output.append(f"{section_num} → {doc_path}")
        outcome = "children"
        summary = f"{len(tree)} children at {relative_path}"

    result = "\n".join(output)
    logger.info(
        "discover_documents done",
        extra={
            "extra_fields": {
                "product_number": product_number,
                "outcome": outcome,
                "summary": summary,
            }
        },
    )
    return result
