<template>
    <div>
      <div v-for="(message, index) in messages" :key="index">
        {{ message }}
      </div>
      <input v-model="inputMessage" @keyup.enter="sendMessage" placeholder="Enter a message">
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        inputMessage: '',
        messages: []
      }
    },
    methods: {
      sendMessage() {
        this.messages.push(this.inputMessage)
        this.inputMessage = ''
        this.messages.push('Response')
      }
    }
  }
  </script>