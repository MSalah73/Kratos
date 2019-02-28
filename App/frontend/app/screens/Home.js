import React from 'react';
import { View, StatusBar, ScrollView } from 'react-native';
import {Container} from '../components/Container'
import {Logo} from '../components/Logo';
import {ImageUploader} from '../components/ImageUploader';
import {ModifiedStatusBar} from '../components/ModifiedStatusBar'

export default () => (
	<Container>
		<ModifiedStatusBar barStyle='light-content'/>
		<ScrollView width='100%' contentContainerStyle={{flexGrow: 1, justifyContent:'center'}}>
			<Logo />
			<View />
			<ImageUploader />
		</ScrollView>
	</Container>
);
