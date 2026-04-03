import { DefaultChatView } from '@/components/workflows/discovery/DefaultChatView';
import { motion } from 'framer-motion';

export default function Dashboard() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="min-h-screen w-full bg-background"
    >
      <DefaultChatView />
    </motion.div>
  );
}
