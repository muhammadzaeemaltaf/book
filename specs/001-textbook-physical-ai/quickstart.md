# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Overview
This guide provides a rapid path to get started with the Physical AI & Humanoid Robotics textbook project. It covers the essential setup steps needed to access, build, and contribute to the textbook content.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended) or any modern Linux distribution
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 20GB free space for development environment
- **Node.js**: Version 18+ (for Docusaurus)
- **Git**: Version control system

### Software Dependencies
- Git: `sudo apt install git`
- Node.js: Download from nodejs.org or use NodeSource
- npm or yarn: Package manager (comes with Node.js)

## Setting Up the Textbook Locally

### 1. Clone the Repository
```bash
git clone https://github.com/[username]/physical-ai-robotics-textbook.git
cd physical-ai-robotics-textbook
```

### 2. Install Dependencies
```bash
npm install
# or if using yarn
yarn install
```

### 3. Start Local Development Server
```bash
npm start
# or
yarn start
```

This will start a local development server at `http://localhost:3000` where you can view the textbook as you work on it.

## Textbook Structure

### Content Organization
```
docs/
├── intro/                    # Introduction module
├── module-01-ros2/          # ROS 2 concepts
├── module-02-digital-twin/  # Simulation concepts
├── module-03-isaac/         # Isaac platform
├── module-04-vla/           # Vision-Language-Action
└── appendices/              # Additional resources
```

### Adding New Content
1. Create a new markdown file in the appropriate module directory
2. Follow the chapter template structure (learning objectives, prerequisites, etc.)
3. Add the new page to `sidebars.js` to make it appear in the navigation

### Content Format
All content follows this standard template:

```markdown
# Chapter Title

## Learning Objectives
- Objective 1 (actionable, measurable)
- Objective 2
- Objective 3

## Prerequisites
- Required knowledge/skills
- Required software/hardware
- Prior chapters to complete

## Introduction
- Why this topic matters
- Real-world applications
- What you'll build by the end

## Core Concepts
### Concept 1
- Explanation with analogy
- Technical details
- Visual diagram

### Concept 2
- Explanation with analogy
- Technical details
- Visual diagram

## Hands-On Tutorial
### Step 1: Setup
- Commands with explanations
- Expected output
- Common issues

### Step 2: Implementation
- Code walkthrough with comments
- Testing procedures
- Validation checks

### Step 3: Extension
- How to customize
- Additional features to try

## Practical Example
- Complete working project
- Step-by-step instructions
- Expected results

## Troubleshooting
- Common Error 1: Cause and solution
- Common Error 2: Cause and solution
- Common Error 3: Cause and solution

## Key Takeaways
- Summary of main concepts
- Skills acquired
- Connection to next chapter

## Additional Resources
- Official documentation links
- Community tutorials
- Advanced reading

## Self-Assessment
- 3-5 questions to verify understanding
- Practical challenges to attempt
```

## Building for Production

### 1. Build Static Site
```bash
npm run build
# or
yarn build
```

This creates a static site in the `build/` directory ready for deployment.

### 2. Serve Built Site Locally
```bash
npm run serve
# or
yarn serve
```

## Contributing to Content

### Content Standards
1. **Technical Accuracy**: All code examples must be tested in Ubuntu 22.04 + ROS 2 Humble
2. **Accessibility**: Use Flesch-Kincaid grade 11-13 writing level
3. **Visual Aids**: Include minimum 2 diagrams per major concept
4. **Hands-on**: Every concept must have practical examples
5. **Links**: All external links must be verified and functional

### Adding Code Examples
1. Place code in the appropriate chapter
2. Test in clean Ubuntu 22.04 + ROS 2 Humble environment
3. Include expected output
4. Add troubleshooting notes for common issues
5. Ensure code block is under 50 lines (split if longer)

### Adding Diagrams
1. Place images in `static/img/` directory
2. Optimize for web delivery (<500KB)
3. Include descriptive alt text for accessibility
4. Reference in markdown using relative paths: `![Description](/img/diagram-name.png)`

## RAG Chatbot Integration

### Local Testing
The textbook includes RAG chatbot integration for enhanced learning experiences. To test locally:

1. Ensure environment variables are set:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export QDRANT_URL=your_qdrant_url
   export QDRANT_API_KEY=your_qdrant_api_key
   ```

2. Run the chatbot backend separately (FastAPI server)
3. The Docusaurus site will connect to the backend for AI-powered Q&A

## Deployment

### GitHub Pages Deployment
The textbook is configured for GitHub Pages deployment:

1. Update `docusaurus.config.js` with your repository details:
   ```javascript
   {
     url: 'https://[username].github.io',
     baseUrl: '/physical-ai-robotics/',
     organizationName: '[username]',
     projectName: 'physical-ai-robotics',
   }
   ```

2. Use the GitHub Actions workflow or deploy manually:
   ```bash
   GIT_USER=[your GitHub username] npm run deploy
   ```

## Testing Your Changes

### Content Validation
Run the content validation scripts to ensure quality:
```bash
# Check for broken links
npm run validate-links

# Check readability score
npm run readability-check

# Test code examples (requires ROS 2 environment)
npm run test-examples
```

### Build Validation
Always test that the site builds successfully:
```bash
npm run build
```

## Common Issues and Solutions

### Build Issues
- **Error**: `Module not found` - Run `npm install` to ensure all dependencies are installed
- **Error**: `Port already in use` - Use `npm start -- --port 3001` to use a different port

### Content Issues
- **Long load times** - Check image sizes and optimize them to <500KB
- **Broken links** - Run link validation script to identify and fix broken links
- **Code examples not working** - Test in clean Ubuntu 22.04 + ROS 2 Humble environment

## Next Steps

1. **Start Reading**: Begin with the Introduction module to understand the learning path
2. **Hands-on Practice**: Follow along with code examples in your own environment
3. **Contribute**: If you find issues or have improvements, submit a pull request
4. **Join Community**: Connect with other learners in the GitHub Discussions

## Support

- **Documentation Issues**: Open an issue in the GitHub repository
- **Technical Questions**: Use GitHub Discussions
- **Content Corrections**: Submit a pull request with fixes
- **Feature Requests**: Open an issue with detailed explanation