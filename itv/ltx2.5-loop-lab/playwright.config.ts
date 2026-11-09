import { defineConfig } from '@playwright/test';
export default defineConfig({testDir:'./tests',timeout:60000,workers:1,use:{baseURL:process.env.LOOPLAB_TEST_URL,channel:'chrome'}});
