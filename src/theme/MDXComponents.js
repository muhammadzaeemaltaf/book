import React from 'react';
import OriginalMDXComponents from '@theme-original/MDXComponents';
import ChapterNavigation from '../components/ChapterNavigation';
import ModuleNavigation from '../components/ModuleNavigation';
import PrerequisiteIndicator from '../components/PrerequisiteIndicator';
import BreadcrumbNavigation from '../components/BreadcrumbNavigation';
import TableOfContents from '../components/TableOfContents';

const MDXComponents = {
  ...OriginalMDXComponents,
  ChapterNavigation,
  ModuleNavigation,
  PrerequisiteIndicator,
  BreadcrumbNavigation,
  TableOfContents,
};

export default MDXComponents;