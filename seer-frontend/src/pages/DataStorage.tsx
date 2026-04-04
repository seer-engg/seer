import { motion } from 'framer-motion';
import { Database } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { KnowledgeBasesCard } from '@/components/data-storage/KnowledgeBasesCard';
import { FilesMediaGallery } from '@/components/data-storage/FilesMediaGallery';
import { EmailAnalyticsTab } from '@/components/data-storage/email-analytics/EmailAnalyticsTab';
import { DatabasesTab } from '@/components/data-storage/DatabasesTab';

export default function DataStorage() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="flex flex-col min-h-screen"
    >
      {/* Header */}
      <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container py-6">
          <div className="flex items-center gap-3 mb-1">
            <div className="p-2 rounded-lg bg-primary/10">
              <Database className="h-5 w-5 text-primary" />
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">Data & Storage</h1>
          </div>
          <p className="text-muted-foreground text-sm">
            Manage your knowledge bases and file storage
          </p>
        </div>
      </div>

      {/* Content */}
      <div className="container py-6 flex-1">
        <Tabs defaultValue="files-media">
          <TabsList>
            <TabsTrigger value="files-media">Files & Media</TabsTrigger>
            <TabsTrigger value="knowledge-bases">Knowledge Bases</TabsTrigger>
            <TabsTrigger value="databases">Databases</TabsTrigger>
            <TabsTrigger value="email-analytics">Email Analytics</TabsTrigger>
          </TabsList>
          <TabsContent value="files-media" className="mt-6">
            <FilesMediaGallery />
          </TabsContent>
          <TabsContent value="knowledge-bases" className="mt-6">
            <KnowledgeBasesCard />
          </TabsContent>
          <TabsContent value="databases" className="mt-6">
            <DatabasesTab />
          </TabsContent>
          <TabsContent value="email-analytics" className="mt-6">
            <EmailAnalyticsTab />
          </TabsContent>
        </Tabs>
      </div>
    </motion.div>
  );
}
