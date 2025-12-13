import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import { ChatWidget } from '../components/ChatWidget';
import { AuthProvider } from '../contexts/AuthContext';

export default function Layout(props) {
  return (
    <AuthProvider>
      <OriginalLayout {...props}>
        {props.children}
        <ChatWidget />
      </OriginalLayout>
    </AuthProvider>
  );
}