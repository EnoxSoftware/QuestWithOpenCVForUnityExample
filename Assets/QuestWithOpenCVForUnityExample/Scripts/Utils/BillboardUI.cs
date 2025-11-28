using UnityEngine;

/// <summary>
/// Billboard UI component that positions and rotates the GameObject to face the user's head (HMD) position.
/// The UI is positioned at a specified distance and vertical/horizontal offset from the head transform.
/// </summary>
public class BillboardUI : MonoBehaviour
{
    // Public Fields
    /// <summary>
    /// Transform representing the user's viewpoint (HMD).
    /// </summary>
    [Tooltip("Transform representing the user's viewpoint (HMD).")]
    public Transform HeadTransform;

    /// <summary>
    /// Distance from the head transform to display the UI in meters.
    /// </summary>
    [Tooltip("Distance from the head transform to display the UI in meters.")]
    public float DistanceFromHead = 1.2f;

    /// <summary>
    /// Vertical offset in meters (up: positive, down: negative).
    /// </summary>
    [Tooltip("Vertical offset in meters (up: positive, down: negative).")]
    public float VerticalOffset = 0f;

    /// <summary>
    /// Horizontal offset in meters (left: negative, right: positive).
    /// </summary>
    [Tooltip("Horizontal offset in meters (left: negative, right: positive).")]
    public float HorizontalOffset = 0f;

    /// <summary>
    /// Follow speed for smooth movement. Higher values mean faster following (recommended: 1-3).
    /// </summary>
    [Tooltip("Follow speed for smooth movement. Higher values mean faster following (recommended: 1-3).")]
    [Range(0.5f, 5f)]
    public float FollowSpeed = 2f;

    /// <summary>
    /// Follow threshold as a ratio of distanceFromHead (0.0 to 1.0).
    /// Billboard will only follow when distance from target exceeds this threshold.
    /// For example, 0.15 means 15% of distanceFromHead.
    /// </summary>
    [Tooltip("Follow threshold as a ratio of distanceFromHead (0.01 to 0.5). Billboard will only follow when distance from target exceeds this threshold.")]
    [Range(0.01f, 0.5f)]
    public float FollowThresholdRatio = 0.30f;

    // Unity Lifecycle Methods
    private void LateUpdate()
    {
        if (HeadTransform == null) return;

        // Calculate target position: head position + forward offset + vertical offset + horizontal offset
        Vector3 targetPos = HeadTransform.position
                            + HeadTransform.forward * DistanceFromHead
                            + Vector3.up * VerticalOffset
                            + HeadTransform.right * HorizontalOffset;

        // Calculate distance from current position to target position
        float distanceToTarget = Vector3.Distance(transform.position, targetPos);

        // Calculate threshold based on distanceFromHead ratio
        float followThreshold = DistanceFromHead * FollowThresholdRatio;

        // Only follow if distance exceeds threshold (allows user to voluntarily look at the UI)
        if (distanceToTarget > followThreshold)
        {
            // Smoothly interpolate position towards target
            float positionLerpFactor = 1f - Mathf.Exp(-FollowSpeed * Time.deltaTime);
            transform.position = Vector3.Lerp(transform.position, targetPos, positionLerpFactor);

            // Calculate target rotation to face the head transform
            Vector3 lookDir = (targetPos - HeadTransform.position).normalized;
            // Restrict to Y-axis rotation only
            lookDir.y = 0f;
            if (lookDir.sqrMagnitude > 0.001f)
            {
                Quaternion targetRotation = Quaternion.LookRotation(lookDir, Vector3.up);
                // Smoothly interpolate rotation towards target
                float rotationLerpFactor = 1f - Mathf.Exp(-FollowSpeed * Time.deltaTime);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, rotationLerpFactor);
            }
        }
    }
}
