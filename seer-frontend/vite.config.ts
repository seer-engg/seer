import { defineConfig, loadEnv, type PluginOption } from "vite";
import react from "@vitejs/plugin-react-swc";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  // Build plugins array with Sentry conditionally included
  const plugins: PluginOption[] = [react()];

  // Add Sentry plugin in production when auth token is available
  if (mode === 'production' && env.SENTRY_AUTH_TOKEN) {
    plugins.push(
      sentryVitePlugin({
        org: env.SENTRY_ORG,
        project: env.SENTRY_PROJECT,
        authToken: env.SENTRY_AUTH_TOKEN,
        release: {
          name: env.VITE_APP_VERSION || `seer-frontend@${Date.now()}`,
        },
        sourcemaps: {
          assets: './dist/**',
          filesToDeleteAfterUpload: './dist/**/*.map',
        },
        telemetry: false,
      }),
    );
  }

  return {
    server: {
      host: "::",
      port: env.VITE_DEV_PORT ? parseInt(env.VITE_DEV_PORT) : 5173,
      strictPort: true,
    },
    plugins,
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    build: {
      sourcemap: true,
    },
  };
});
