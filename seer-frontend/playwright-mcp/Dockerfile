FROM mcr.microsoft.com/playwright:v1.49.1-noble

WORKDIR /app
RUN npm install -g @playwright/mcp@latest && \
    npx playwright install chromium

EXPOSE 9000

CMD ["npx", "@playwright/mcp", "--port", "9000", "--host", "0.0.0.0", "--headless", "--no-sandbox", "--executable-path", "/ms-playwright/chromium-1208/chrome-linux/chrome"]
