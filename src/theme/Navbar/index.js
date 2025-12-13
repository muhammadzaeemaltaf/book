import React from 'react';
import OriginalNavbar from '@theme-original/Navbar';

// We're using Navbar/Content for the auth integration now
// This wrapper just passes through to the original
export default function Navbar(props) {
  return <OriginalNavbar {...props} />;
}