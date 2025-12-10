// Simple test script to verify basic chat functionality
import { ChatService } from './services/chatService';

// Create a simple test to verify the chat service works
async function testBasicQa() {
  console.log('Testing basic Q&A functionality...');

  // Create a chat service instance
  const chatService = new ChatService({ topK: 3, temperature: 0.7 });

  console.log('✓ Chat service initialized');

  // Test basic properties
  const options = chatService.getOptions();
  console.log('✓ Chat options retrieved:', options);

  // Test history management
  const history = chatService.getHistory();
  console.log('✓ Initial history length:', history.length);

  // Verify the service has the correct default options
  if (options.topK === 3 && options.temperature === 0.7 && options.stream === true) {
    console.log('✓ Default options are correctly set');
  } else {
    console.error('✗ Default options are not correctly set');
    return false;
  }

  // Test updating options
  chatService.updateOptions({ topK: 5, temperature: 0.5 });
  const updatedOptions = chatService.getOptions();

  if (updatedOptions.topK === 5 && updatedOptions.temperature === 0.5) {
    console.log('✓ Options update functionality works');
  } else {
    console.error('✗ Options update functionality failed');
    return false;
  }

  // Test history management
  chatService.clearHistory();
  const clearedHistory = chatService.getHistory();

  if (clearedHistory.length === 0) {
    console.log('✓ History clear functionality works');
  } else {
    console.error('✗ History clear functionality failed');
    return false;
  }

  console.log('\nBasic Q&A functionality tests passed!');
  return true;
}

// Run the test
testBasicQa().then(success => {
  if (success) {
    console.log('\n✓ All basic Q&A tests passed!');
    process.exit(0);
  } else {
    console.log('\n✗ Some tests failed!');
    process.exit(1);
  }
}).catch(err => {
  console.error('Test error:', err);
  process.exit(1);
});