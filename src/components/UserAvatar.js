import React from "react";
import styles from "./UserAvatar.module.css";

/**
 * UserAvatar component displays a user's avatar with their initials
 * @param {Object} props
 * @param {Object} props.user - User object containing name, email, and id
 * @param {number} [props.size=32] - Size of the avatar in pixels
 * @param {boolean} [props.showName=false] - Whether to show the user's name next to avatar
 * @param {function} [props.onClick] - Click handler for the avatar
 */
const UserAvatar = ({ user, size = 32, showName = false, onClick }) => {
  if (!user) return null;

  // Generate initials from name or email
  const getInitials = (name, email) => {
    if (name) {
      return name
        .split(" ")
        .map((n) => n[0])
        .join("")
        .toUpperCase()
        .slice(0, 2);
    }
    if (email) {
      return email.charAt(0).toUpperCase();
    }
    return "?";
  };

  // Generate consistent color based on user ID
  const getAvatarColor = (id) => {
    if (!id) return "#6b7280";

    let hash = 0;
    const idStr = String(id);
    for (let i = 0; i < idStr.length; i++) {
      hash = idStr.charCodeAt(i) + ((hash << 5) - hash);
    }

    const colors = [
      "#3b82f6", // blue
      "#10b981", // green
      "#8b5cf6", // purple
      "#f59e0b", // amber
      "#ef4444", // red
      "#6366f1", // indigo
      "#ec4899", // pink
      "#14b8a6", // teal
    ];

    return colors[Math.abs(hash) % colors.length];
  };

  const initials = getInitials(user.name, user.email);
  const backgroundColor = getAvatarColor(user.id);
  const displayName = user.name || user.email?.split("@")[0] || "User";

  const avatarStyle = {
    width: `${size}px`,
    height: `${size}px`,
    backgroundColor,
    fontSize: `${size * 0.4}px`,
  };

  return (
    <div
      className={`${styles.userAvatar} ${onClick ? styles.clickable : ""}`}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className={styles.avatar} style={avatarStyle}>
        {initials}
      </div>
      {showName && <span className={styles.userName}>{displayName}</span>}
    </div>
  );
};

export default UserAvatar;
