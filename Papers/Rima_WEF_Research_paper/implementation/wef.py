import torch


def initialize_wef(weight: torch.Tensor) -> torch.Tensor:
    """
    Create an all-zero WEF-Matrix with the same shape
    as the selected model weight.
    """

    return torch.zeros_like(
        weight.detach().cpu(),
        dtype=torch.float32,
    )


def update_wef_matrix(
    wef_matrix: torch.Tensor,
    previous_weight: torch.Tensor,
    current_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Update WEF-Matrix.

    1. Calculate absolute weight change.
    2. Calculate average change as a dynamic threshold.
    3. Increase WEF value where change is greater
       than the threshold.
    """

    difference = torch.abs(
        current_weight.detach().cpu()
        - previous_weight.detach().cpu()
    )

    threshold = difference.mean()

    significant_changes = (
        difference > threshold
    ).to(torch.float32)

    return wef_matrix + significant_changes
