import React from 'react';
import { Redirect } from '@docusaurus/router';

// Redirect auth root to signin page
const AuthIndex = () => {
  return <Redirect to="/signin" />;
};

export default AuthIndex;